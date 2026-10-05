"""Read-only JSON API over the precomputed `projections` table.

    python -m uvicorn api.main:app --reload                      # local, http://127.0.0.1:8000/docs
    uvicorn api.main:app --host 0.0.0.0 --port $PORT              # deployed

Environment:
    DATABASE_URL_POOLED / DATABASE_URL   Postgres; the pooled URL is preferred
    CORS_ORIGINS        comma-separated allowed origins, default "*"
    CORS_ORIGIN_REGEX   optional, e.g. https://my-app(-[a-z0-9-]+)?\\.vercel\\.app for previews
    SEASON_WEEK1_THURSDAY   Thursday of week 1, default 2026-09-10; sets the default week

Every error is {"error": {"code", "message", "details"}}: 400 for bad input,
404 for an unknown player or a week with no projections, 409 when a name
matches several players, 503 when the database cannot be reached.
"""

import os
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import db, names

POSITIONS = ("QB", "RB", "WR", "TE")
FIRST_WEEK, LAST_WEEK = 1, 18
MAX_LIMIT = 100
# The NFL day is US time; a UTC server would roll the default week over hours early.
NFL_TZ = timezone(timedelta(hours=-5))

app = FastAPI(
    title="Fantasy Football Projections API",
    version="1.0.0",
    description="Weekly PPR projections for 2026 QBs, RBs, WRs and TEs, precomputed by batch_compute.py.",
)


def _origins() -> list[str]:
    return [origin.strip() for origin in os.environ.get("CORS_ORIGINS", "*").split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX") or None,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        self.status, self.code, self.message, self.details = status, code, message, details or {}


def _error_response(status: int, code: str, message: str, details: dict | None = None,
                    headers: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"code": code, "message": message, "details": details or {}}})


@app.exception_handler(ApiError)
async def _api_error(_: Request, error: ApiError):
    return _error_response(error.status, error.code, error.message, error.details)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, error: RequestValidationError):
    problems = [{"parameter": ".".join(str(part) for part in e["loc"][1:]) or str(e["loc"][0]),
                 "message": e["msg"]} for e in error.errors()]
    message = "; ".join(f"{p['parameter']}: {p['message']}" for p in problems)
    return _error_response(400, "invalid_parameter", message, {"problems": problems})


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, error: StarletteHTTPException):
    code = "not_found" if error.status_code == 404 else "http_error"
    return _error_response(error.status_code, code, str(error.detail))


@app.exception_handler(db.DatabaseUnavailable)
async def _database_unavailable(_: Request, error: db.DatabaseUnavailable):
    return _error_response(503, "database_unavailable",
                           "The projections database is waking up or unreachable; retry shortly.",
                           headers={"Retry-After": "5"})


@app.exception_handler(Exception)
async def _unexpected(_: Request, error: Exception):
    return _error_response(500, "internal_error", "Unexpected server error.")


# ---------------------------------------------------------------------------
# Response shapes
# ---------------------------------------------------------------------------

class Player(BaseModel):
    player_id: str
    player_name: str
    position: str
    team: str


class Stats(BaseModel):
    passing_attempts: float | None
    passing_yards: float | None
    passing_tds: float | None
    interceptions: float | None
    rushing_yards: float | None
    rushing_tds: float | None
    carries: float | None
    receptions: float | None
    receiving_yards: float | None
    receiving_tds: float | None


class Projection(BaseModel):
    player: Player
    week: int
    opponent: str
    projected_ppr: float
    stats: Stats
    computed_at: datetime


class Resolved(BaseModel):
    query: str
    matched_by: Literal["player_id", "exact", "nickname", "last_name"]


class PlayerProjectionResponse(Projection):
    resolved: Resolved
    available_weeks: list[int]


class RankedProjection(Projection):
    rank: int


class RankingsResponse(BaseModel):
    position: str
    week: int
    limit: int
    count: int
    rankings: list[RankedProjection]


class PlayersResponse(BaseModel):
    count: int
    players: list[Player]


class WeeksResponse(BaseModel):
    weeks: list[int]
    default_week: int | None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict


class ErrorResponse(BaseModel):
    error: ErrorBody


ERRORS = {status: {"model": ErrorResponse} for status in (400, 404, 409, 503)}


def _round(value):
    return None if value is None else round(value, 2)


def _projection(row: dict) -> dict:
    return {
        "player": {key: row[key] for key in ("player_id", "player_name", "position", "team")},
        "week": row["week"],
        "opponent": row["opponent"],
        "projected_ppr": _round(row["projected_ppr"]),
        "stats": {column: _round(row[column]) for column in db.STAT_COLUMNS},
        "computed_at": row["computed_at"],
    }


# ---------------------------------------------------------------------------
# Weeks
# ---------------------------------------------------------------------------

def upcoming_week(today: date | None = None) -> int:
    """The first regular-season week none of whose games has kicked off.

    Same rule as batch_compute.remaining_weeks, but from the calendar instead of
    the schedule file, which the deployed API doesn't have: week N starts on the
    Thursday N-1 weeks after week 1's.
    """
    today = today or datetime.now(NFL_TZ).date()
    week1 = date.fromisoformat(os.environ.get("SEASON_WEEK1_THURSDAY", "2026-09-10"))
    return min(LAST_WEEK + 1, max(FIRST_WEEK, (today - week1).days // 7 + 2))


def default_week(weeks: list[int]) -> int | None:
    """The upcoming week, or the nearest later one with projections; the last if the season is over."""
    if not weeks:
        return None
    later = [week for week in weeks if week >= upcoming_week()]
    return later[0] if later else weeks[-1]


WeekParam = Query(None, ge=FIRST_WEEK, le=LAST_WEEK,
                  description="Regular-season week (1-18). Defaults to the next upcoming week.")


def _week_or_default(week: int | None, weeks: list[int]) -> int:
    if week is None:
        week = default_week(weeks)
        if week is None:
            raise ApiError(404, "no_projections", "No projections have been computed yet.")
    if week not in weeks:
        raise ApiError(404, "week_not_found", f"No projections for week {week}.",
                       {"week": week, "available_weeks": weeks})
    return week


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    """Liveness plus a database round trip (which also wakes Neon)."""
    db.fetch("SELECT 1")
    return {"status": "ok"}


@app.get("/weeks", response_model=WeeksResponse, responses=ERRORS, tags=["meta"])
def weeks():
    """Weeks that have projections, and the week endpoints use when none is given."""
    available = db.available_weeks()
    return {"weeks": available, "default_week": default_week(available)}


@app.get("/players", response_model=PlayersResponse, responses=ERRORS, tags=["players"])
def list_players(
    position: Literal["QB", "RB", "WR", "TE"] | None = Query(None, description="Only this position."),
    team: str | None = Query(None, min_length=2, max_length=3, description="Only this team, e.g. MIN."),
    q: str | None = Query(None, min_length=1, max_length=60,
                          description="Case-insensitive substring of the player name, for autocomplete."),
):
    """Every projected player, sorted by name, with position and team to tell namesakes apart."""
    rows = db.players()
    if position:
        rows = [r for r in rows if r["position"] == position]
    if team:
        rows = [r for r in rows if r["team"] == team.upper()]
    if q:
        needle = names.normalize(q)
        rows = [r for r in rows if needle in names.normalize(r["player_name"])]
    rows.sort(key=lambda r: (r["player_name"], r["player_id"]))
    return {"count": len(rows), "players": rows}


@app.get("/projections/player", response_model=PlayerProjectionResponse, responses=ERRORS,
         tags=["projections"])
def player_projection(
    player_id: str | None = Query(None, min_length=1, max_length=20,
                                  description="GSIS id, e.g. 00-0036322. Preferred over name."),
    name: str | None = Query(None, min_length=1, max_length=60,
                             description="Player name; tolerant of case, punctuation, suffixes and nicknames."),
    position: Literal["QB", "RB", "WR", "TE"] | None = Query(None, description="Narrows a name lookup."),
    team: str | None = Query(None, min_length=2, max_length=3, description="Narrows a name lookup."),
    week: int | None = WeekParam,
):
    """One player's projection for a week. Give player_id, or name (optionally with position/team).

    A name that matches several players answers 409 with the candidates; retry with their player_id.
    """
    if (player_id is None) == (name is None):
        raise ApiError(400, "invalid_parameter", "Give exactly one of player_id or name.")

    if player_id is not None:
        query, matched_by = player_id, "player_id"
    else:
        resolution = names.resolve(name, db.players(), position, team.upper() if team else None)
        if not resolution.matches:
            raise ApiError(404, "player_not_found", f"No projected player matches '{name}'.",
                           {"query": name, "suggestions": resolution.suggestions})
        if len(resolution.matches) > 1:
            raise ApiError(409, "ambiguous_player",
                           f"'{name}' matches {len(resolution.matches)} players; retry with a player_id.",
                           {"query": name, "matched_by": resolution.matched_by,
                            "candidates": sorted(resolution.matches, key=lambda p: p["player_name"])})
        query, matched_by = name, resolution.matched_by
        player_id = resolution.matches[0]["player_id"]

    rows = db.player_week_rows(player_id)
    if not rows:
        raise ApiError(404, "player_not_found", f"No projections for player_id '{player_id}'.",
                       {"player_id": player_id})
    player_weeks = [row["week"] for row in rows]

    if week is None:
        week = default_week(db.available_weeks())
    if week not in player_weeks:
        # A week other players have is this player's bye (or he wasn't projected).
        all_weeks = db.available_weeks()
        reason = "is a bye or unprojected week for this player" if week in all_weeks else "has no projections"
        raise ApiError(404, "week_not_found", f"Week {week} {reason}.",
                       {"week": week, "player_id": player_id, "available_weeks": player_weeks})

    row = next(row for row in rows if row["week"] == week)
    return {**_projection(row), "resolved": {"query": query, "matched_by": matched_by},
            "available_weeks": player_weeks}


@app.get("/projections/rankings", response_model=RankingsResponse, responses=ERRORS,
         tags=["projections"])
def position_rankings(
    position: Literal["QB", "RB", "WR", "TE"] = Query(..., description="QB, RB, WR or TE."),
    week: int | None = WeekParam,
    limit: int = Query(10, ge=1, le=MAX_LIMIT, description=f"How many players, 1-{MAX_LIMIT}."),
):
    """The top players at a position for a week, by projected PPR."""
    week = _week_or_default(week, db.available_weeks())
    rows = db.rankings(position, week, limit)
    rankings = [{**_projection(row), "rank": rank} for rank, row in enumerate(rows, start=1)]
    return {"position": position, "week": week, "limit": limit, "count": len(rankings), "rankings": rankings}
