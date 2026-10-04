"""Project every 2026 player for every remaining week and write the lines to Postgres.

    python batch_compute.py                 # every week no game of which has kicked off
    python batch_compute.py --weeks 5 6     # specific weeks
    python batch_compute.py --dry-run       # compute only, skip the database

Every projection is prediction.predict() itself, unchanged. What a single run of
it spends its ~1.7s on is reloading the same parquet, CSVs, models and season
form for each player, so for the batch those loaders are swapped for cached ones
while predict runs. Two lookups are also pinned: the player is found by
player_id rather than by name (training and roster names differ for a dozen
players, e.g. "Kenneth Gainwell" / "Kenny Gainwell"), and the stat line predict
prints is captured as numbers instead.

The players are the 2026 QBs, RBs, WRs and TEs in final_training_data.parquet.
Rows land in the `projections` table keyed on (player_name, week); a run replaces
every row of the weeks it projects.
"""

import argparse
import contextlib
import io
import os
import time
from datetime import date
from unittest import mock

import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv
from psycopg2.extras import execute_values

import prediction

POSITIONS = ("QB", "RB", "WR", "TE")
STAT_COLUMNS = ("passing_attempts", "passing_yards", "passing_tds", "interceptions",
                "rushing_yards", "rushing_tds", "carries", "receptions",
                "receiving_yards", "receiving_tds")
PROGRESS_EVERY = 25

CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS projections (
    player_name     text             NOT NULL,
    player_id       text,
    position        text             NOT NULL,
    team            text             NOT NULL,
    week            integer          NOT NULL,
    opponent        text             NOT NULL,
    projected_ppr   double precision NOT NULL,
    {", ".join(f"{column} double precision" for column in STAT_COLUMNS)},
    computed_at     timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (player_name, week)
)
"""

COLUMNS = ("player_name", "player_id", "position", "team", "week", "opponent",
           "projected_ppr") + STAT_COLUMNS


# ---------------------------------------------------------------------------
# What to project
# ---------------------------------------------------------------------------

def remaining_weeks(today: date) -> list[int]:
    """Regular-season weeks none of whose games has kicked off yet."""
    schedule = prediction.load_csv(prediction.SCHEDULE_PATH, f"{prediction.TARGET_SEASON} schedule")
    first_game = prediction.regular_season(schedule).groupby("week")["gameday"].min()
    return [int(week) for week, day in first_game.items() if day.date() > today]


def load_players() -> pd.DataFrame:
    """One row per 2026 skill player: player_id and the name training data knows him by."""
    rows = pd.read_parquet(prediction.TRAINING_PATH,
                           columns=["season", "week", "player_id", "player_name", "bio_position"],
                           filters=[("season", "==", float(prediction.TARGET_SEASON))])
    rows = rows[rows["bio_position"].isin(POSITIONS) & rows["player_id"].notna()]
    latest = rows.sort_values("week").groupby("player_id").tail(1)
    return latest.sort_values("player_name")[["player_id", "player_name"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Running prediction.predict in bulk
# ---------------------------------------------------------------------------

class BatchPredictor:
    """Runs prediction.predict with its loaders cached and its result captured."""

    def __init__(self):
        self._json, self._csv, self._parquet, self._models, self._forms = {}, {}, {}, {}, {}
        self._originals = {
            "load_json": prediction.load_json,
            "load_csv": prediction.load_csv,
            "load_models": prediction.load_models,
            "SeasonForm": prediction.SeasonForm,
            "find_player": prediction.find_player,
            "find_game": prediction.find_game,
            "format_detail": prediction.format_detail,
            "read_parquet": pd.read_parquet,
        }
        self.metadata = prediction.load_json(prediction.METADATA_PATH)
        self._reset(None)

    def _reset(self, player_id) -> None:
        self.player_id = player_id
        self.player = self.team = self.game = self.is_home = None
        self.position = self.predictions = None

    # -- cached loaders -----------------------------------------------------

    def _load_json(self, path):
        if path not in self._json:
            self._json[path] = self._originals["load_json"](path)
        return self._json[path]

    def _load_csv(self, path, label):
        if path not in self._csv:
            self._csv[path] = self._originals["load_csv"](path, label)
        return self._csv[path]

    def _read_parquet(self, path, columns=None, filters=None, **kwargs):
        key = (str(path), tuple(columns) if columns is not None else None, repr(filters), repr(kwargs))
        if key not in self._parquet:
            self._parquet[key] = self._originals["read_parquet"](path, columns=columns, filters=filters, **kwargs)
        return self._parquet[key]

    def _load_models(self, position_meta):
        key = tuple(sorted(info["model_path"] for info in position_meta.values()))
        if key not in self._models:
            self._models[key] = self._originals["load_models"](position_meta)
        return self._models[key]

    def _season_form(self, target_week, schedule):
        # SeasonForm depends only on the target week; every lookup on it is read-only.
        if target_week not in self._forms:
            self._forms[target_week] = self._originals["SeasonForm"](target_week, schedule)
        return self._forms[target_week]

    # -- pinned lookups and captured output ---------------------------------

    def _find_player(self, bio, name):
        pinned = bio[bio["player_id"] == self.player_id]
        if pinned.empty:
            prediction.fail(f"player_id {self.player_id} not on the {prediction.TARGET_SEASON} rosters")
        self.player = self._originals["find_player"](pinned, name)
        return self.player

    def _find_game(self, schedule, team, week):
        self.team = team
        self.game, self.is_home = self._originals["find_game"](schedule, team, week)
        return self.game, self.is_home

    def _format_detail(self, position, predictions):
        self.position, self.predictions = position, dict(predictions)
        return self._originals["format_detail"](position, predictions)

    @contextlib.contextmanager
    def patched(self):
        with contextlib.ExitStack() as stack:
            for attribute, replacement in (("load_json", self._load_json), ("load_csv", self._load_csv),
                                           ("load_models", self._load_models),
                                           ("SeasonForm", self._season_form),
                                           ("find_player", self._find_player), ("find_game", self._find_game),
                                           ("format_detail", self._format_detail),
                                           ("refresh_depth_chart", lambda: None)):
                stack.enter_context(mock.patch.object(prediction, attribute, replacement))
            stack.enter_context(mock.patch.object(pd, "read_parquet", self._read_parquet))
            yield

    def roster_name(self, player_id) -> str | None:
        bio = self._load_csv(prediction.BIO_PATH, f"{prediction.TARGET_SEASON} bio data")
        rows = bio[bio["player_id"] == player_id]
        return None if rows.empty else str(rows.sort_values("week")["player_name"].iloc[-1])

    def project(self, player_id, name: str, week: int) -> dict:
        """One player-week, exactly as `python prediction.py NAME --week WEEK` computes it."""
        self._reset(player_id)
        with contextlib.redirect_stdout(io.StringIO()):
            prediction.predict(name, week)

        weights = self.metadata["ppr_weights"]
        points = max(0.0, sum(self.predictions.get(stat, 0.0) * weight for stat, weight in weights.items()))
        opponent = self.game["away_team"] if self.is_home else self.game["home_team"]
        row = {
            "player_id": player_id,
            "player_name": self.player["player_name"],
            "position": self.position,
            "team": self.team,
            "week": week,
            "opponent": prediction.normalize_team(opponent),
            "projected_ppr": points,
        }
        row.update({column: self.predictions.get(column) for column in STAT_COLUMNS})
        return row


def compute(players: pd.DataFrame, weeks: list[int]) -> tuple[list[dict], list[str], int]:
    predictor = BatchPredictor()
    rows, errors, byes = [], [], 0
    started = time.time()

    with predictor.patched():
        for done, player in enumerate(players.itertuples(index=False), start=1):
            roster_name = predictor.roster_name(player.player_id)
            if roster_name is None:
                errors.append(f"{player.player_name} ({player.player_id}): not on the "
                              f"{prediction.TARGET_SEASON} rosters")
            else:
                for week in weeks:
                    try:
                        row = predictor.project(player.player_id, roster_name, week)
                    except prediction.PredictionError as exc:
                        if "bye week" in str(exc):
                            byes += 1
                            continue
                        errors.append(f"{player.player_name} week {week}: {exc}")
                        if "models exist only for" in str(exc):
                            break
                        continue
                    except Exception as exc:
                        errors.append(f"{player.player_name} week {week}: {type(exc).__name__}: {exc}")
                        continue
                    rows.append(row)

            if done % PROGRESS_EVERY == 0 or done == len(players):
                elapsed = time.time() - started
                eta = elapsed / done * (len(players) - done)
                print(f"  {done}/{len(players)} players | {len(rows)} rows, {byes} byes, "
                      f"{len(errors)} errors | {elapsed:.0f}s elapsed, ~{eta:.0f}s left", flush=True)

    return rows, errors, byes


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------

def database_engine() -> sa.Engine:
    load_dotenv(prediction.PROJECT_DIR / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set; add it to .env")
    # SQLAlchemy only accepts the postgresql:// spelling.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return sa.create_engine(url)


def write_rows(engine: sa.Engine, rows: list[dict], weeks: list[int], replace_weeks: bool) -> None:
    """Write in one transaction, so readers never see a half-written week.

    With replace_weeks, every existing row of the projected weeks is dropped
    first; a partial run (--limit) only upserts its own players.
    """
    values = [tuple(row[column] for column in COLUMNS) for row in rows]
    updates = ", ".join(f"{column} = EXCLUDED.{column}" for column in COLUMNS
                        if column not in ("player_name", "week"))
    connection = engine.raw_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_TABLE)
            # A player cut or retired since the last run has no row this time;
            # his old projection for these weeks should not outlive him.
            if replace_weeks:
                cursor.execute("DELETE FROM projections WHERE week = ANY(%s)", (weeks,))
            execute_values(
                cursor,
                f"INSERT INTO projections ({', '.join(COLUMNS)}) VALUES %s "
                f"ON CONFLICT (player_name, week) DO UPDATE SET {updates}, computed_at = now()",
                values, page_size=1000)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Project all 2026 players for the remaining weeks into Postgres.")
    parser.add_argument("--weeks", type=int, nargs="+", metavar="N",
                        help="weeks to project (default: every week with no game kicked off)")
    parser.add_argument("--limit", type=int, metavar="N", help="only the first N players, for testing")
    parser.add_argument("--dry-run", action="store_true", help="compute but do not write to the database")
    parser.add_argument("--no-refresh", action="store_true",
                        help=f"use the saved depth chart even if it is over {prediction.DEPTH_REFRESH_HOURS}h old")
    args = parser.parse_args()

    today = date.today()
    weeks = sorted(args.weeks) if args.weeks else remaining_weeks(today)
    if not weeks:
        raise SystemExit(f"no {prediction.TARGET_SEASON} weeks left to project as of {today}")

    players = load_players()
    if args.limit:
        players = players.head(args.limit)

    engine = None
    if not args.dry_run:
        # Fail on a bad connection before spending minutes on projections.
        engine = database_engine()
        with engine.connect() as connection:
            connection.execute(sa.text("SELECT 1"))

    # Refreshed once up front instead of being checked inside every predict call.
    if args.no_refresh:
        prediction.REFRESH_DEPTH_CHART = False
    refreshed = prediction.refresh_depth_chart()
    if refreshed:
        print(f"depth chart refresh: {refreshed}")

    print(f"Projecting {len(players)} players x weeks {weeks[0]}-{weeks[-1]} ({len(weeks)} weeks) as of {today}")
    started = time.time()
    rows, errors, byes = compute(players, weeks)
    print(f"Computed {len(rows)} projections ({byes} bye weeks skipped) in {time.time() - started:.0f}s")

    if errors:
        print(f"{len(errors)} error(s):")
        for error in errors:
            print(f"  {error}")

    keys = pd.DataFrame(rows, columns=list(COLUMNS))[["player_name", "week"]]
    duplicates = keys[keys.duplicated(keep=False)]
    if not duplicates.empty:
        raise SystemExit(f"two players share a name, which the (player_name, week) key cannot hold:\n"
                         f"{duplicates.drop_duplicates().to_string(index=False)}")

    if args.dry_run:
        print("Dry run: nothing written.")
        return
    if not rows:
        raise SystemExit("nothing to write")

    started = time.time()
    write_rows(engine, rows, weeks, replace_weeks=not args.limit)
    print(f"Wrote {len(rows)} rows to projections in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
