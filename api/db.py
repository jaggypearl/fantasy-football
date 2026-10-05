"""Read-only access to the `projections` table that batch_compute.py fills.

Neon scales to zero when idle, so the first query after a quiet spell can wait
on a cold start or hit a connection the server already dropped. Connections are
pinged before use and recycled well inside Neon's idle timeout, and a query
that still fails on a connection-level error is retried with backoff before
the API answers 503.
"""

import os
import time
from pathlib import Path

import sqlalchemy as sa
from dotenv import load_dotenv

STAT_COLUMNS = ("passing_attempts", "passing_yards", "passing_tds", "interceptions",
                "rushing_yards", "rushing_tds", "carries", "receptions",
                "receiving_yards", "receiving_tds")

RETRY_DELAYS = (0.5, 1.5, 3.0)


class DatabaseUnavailable(Exception):
    """The database could not be reached after every retry."""


def database_url() -> str:
    # A local .env is a convenience for development; deployed, the variables come
    # from the host's environment and nothing here overrides them.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
    url = os.environ.get("DATABASE_URL_POOLED") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATABASE_URL_POOLED or DATABASE_URL")
    # SQLAlchemy only accepts the postgresql:// spelling.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def create_engine() -> sa.Engine:
    return sa.create_engine(
        database_url(),
        pool_pre_ping=True,
        pool_recycle=240,
        pool_size=5,
        max_overflow=5,
        connect_args={"connect_timeout": int(os.environ.get("DB_CONNECT_TIMEOUT", "15"))},
    )


_engine: sa.Engine | None = None


def engine() -> sa.Engine:
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def fetch(sql: str, **params) -> list[dict]:
    """Run a read query and return its rows as dicts, retrying through cold starts."""
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        try:
            with engine().connect() as connection:
                return [dict(row) for row in connection.execute(sa.text(sql), params).mappings()]
        except (sa.exc.OperationalError, sa.exc.InterfaceError) as error:
            # Connection-level failures only; a bad query raises ProgrammingError
            # and should surface, not be retried.
            if delay is None:
                raise DatabaseUnavailable(str(error.orig or error)) from error
            engine().dispose()
            time.sleep(delay)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

PROJECTION_COLUMNS = ", ".join(("player_id", "player_name", "position", "team", "week", "opponent",
                                "projected_ppr", *STAT_COLUMNS, "computed_at"))


def players() -> list[dict]:
    """Every projected player once, with the team of his latest projected week."""
    return fetch("""
        SELECT DISTINCT ON (player_id) player_id, player_name, position, team
        FROM projections
        ORDER BY player_id, week DESC
    """)


def available_weeks() -> list[int]:
    return [row["week"] for row in fetch("SELECT DISTINCT week FROM projections ORDER BY week")]


def player_week_rows(player_id: str) -> list[dict]:
    return fetch(f"SELECT {PROJECTION_COLUMNS} FROM projections WHERE player_id = :player_id ORDER BY week",
                 player_id=player_id)


def rankings(position: str, week: int, limit: int) -> list[dict]:
    return fetch(f"""
        SELECT {PROJECTION_COLUMNS} FROM projections
        WHERE position = :position AND week = :week
        ORDER BY projected_ppr DESC, player_name
        LIMIT :limit
    """, position=position, week=week, limit=limit)
