"""Project a player's PPR line for a given week of the 2026 season.

    python prediction.py "Ja'Marr Chase"              # next unplayed week
    python prediction.py "Ja'Marr Chase" --week 7     # a specific week

With no --week, the upcoming week is read off the 2026 schedule: the first
regular-season week that still has an unplayed game, skipping past weeks whose
scores simply have not landed in the feed yet.

The models are trained through 2025, so a player's baseline feature vector still
comes from their 2025 game log. Anything the 2026 season has actually settled by
kickoff of the target week -- snap-adjacent usage, target share, depth chart,
o-line form, injuries, the game's betting line -- is layered on top of that
baseline, so a Week 7 projection is not a Week 1 projection with the week number
changed. See SEASON_FORM_NOTES for what is and is not refreshed.
"""

import argparse
import itertools
import json
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from processing.layer3_player_features import (
    MIN_CARRIES_FOR_YPC,
    MIN_TARGETS_FOR_RATE,
    REC_RATE_SHRINKAGE_GAMES,
    REC_VOLUME_SHRINKAGE_GAMES,
    RECENT_GAMES,
    RUSH_VOLUME_SHRINKAGE_GAMES,
    RUSH_YARDS_SHRINKAGE_GAMES,
    TEAM_PASS_SHRINKAGE_GAMES,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

TARGET_SEASON = 2026
CONTEXT_SEASON = 2025

METADATA_PATH = MODEL_DIR / "model_metadata.json"
SCHEDULE_PATH = RAW_DIR / "schedules" / f"schedules_{TARGET_SEASON}.csv"
BIO_PATH = RAW_DIR / "bio_data" / f"bio_data_{TARGET_SEASON}.csv"
OC_PATH = RAW_DIR / "offensive_coordinators" / f"offensive_coordinators_{TARGET_SEASON}.csv"
SOS_PATH = RAW_DIR / "strength_of_schedule" / f"strength_of_schedule_{TARGET_SEASON}.csv"
DEPTH_CHART_PATH = RAW_DIR / "depth_chart" / f"depth_chart_{TARGET_SEASON}.csv"
BOX_SCORE_PATH = RAW_DIR / "box_scores" / f"box_scores_{TARGET_SEASON}.csv"
PRIOR_BOX_SCORE_PATH = RAW_DIR / "box_scores" / f"box_scores_{CONTEXT_SEASON}.csv"
INJURY_PATH = RAW_DIR / "injury_reports" / f"injury_reports_{TARGET_SEASON}.csv"
OLINE_PATH = RAW_DIR / "oline_rankings_weekly" / f"oline_rankings_weekly_{TARGET_SEASON}.csv"
PRIOR_OLINE_PATH = RAW_DIR / "oline_rankings_weekly" / f"oline_rankings_weekly_{CONTEXT_SEASON}.csv"
TEAM_AGGS_PATH = PROCESSED_DIR / "team_aggs.parquet"
TRAINING_PATH = PROCESSED_DIR / "final_training_data.parquet"

DEFAULT_WEATHER_TEMP = 70.0
DEFAULT_WEATHER_WIND = 5.0
DEFAULT_SPREAD_LINE = 0.0
DEFAULT_TOTAL_LINE = 45.0
DEPTH_SCORE_UNKNOWN = 0.7
NO_INJURY_WEEKS = 99.0

# Statuses Layer 1 treats as a did-not-play when it builds dnp_flag.
DNP_STATUSES = ("Out", "IR")

# In-season form is shrunk toward the prior-season baseline with weight
# n / (n + SHRINKAGE_GAMES), n being games played this season: 1 game counts a
# third, 2 games half, 6 games three quarters. 2 is the constant that best
# predicted rest-of-season target share from the first 1-6 games of 2016-2025,
# at every position, beating both the raw in-season share and the prior season.
SHRINKAGE_GAMES = 2.0

# The trend ratios divide an n-game sample by last season's same n-game span,
# so both sides are noisy; they shrink toward 1.0 with 6 in place of 2 (2 games
# count a quarter). 6 was the best fit for rest-of-season usage and share
# change on the same 2016-2025 games.
TREND_SHRINKAGE_GAMES = 6.0

# Layer 3 clips the trend ratios to this floor.
TREND_FLOOR = 0.1

# The depth chart feed publishes a new snapshot about twice a day. A local copy
# older than this is re-fetched before projecting, so the label the models see
# is today's, not the one from last week's kickoff.
DEPTH_REFRESH_HOURS = 12
REFRESH_DEPTH_CHART = True

# The positions whose projections re-weight draft pedigree and depth chart role
# below. The models themselves are untouched: both are handled by averaging
# model outputs over alternative values of the feature (see mixture_predict).
ROLE_BLEND_POSITIONS = ("WR", "RB")

# Draft slot is a talent prior for a player with no NFL tape. Each season in
# the league cuts its pull to a quarter of the season before: full weight as a
# rookie, 25% in year two, 6% in year three. The remaining weight goes to the
# position's spread of draft slots, so a veteran is projected as if his draft
# slot were unknown and his game log has to carry him.
DRAFT_WEIGHT_DECAY = 0.25
DRAFT_REFERENCE_QUANTILES = np.linspace(0.05, 0.95, 10)

# The depth chart label is blended with the role this season's usage implies
# (rank in targets per game among the team's players at the position, carries
# plus targets for backs). The label keeps weight
#     DEPTH_CHART_FLOOR + (1 - DEPTH_CHART_FLOOR) * k / (k + n)
# with n games played: all of it before week 1, 68% after one game, 47% after
# three, 40% after six, never below 25%. k = 1.5 was fit on 2025 (the first
# season in the snapshot chart schema 2026 uses) by regressing rest-of-season
# PPR per game on the tier means of each signal after n games; the floor was
# fixed at 0.25, where the 2016-2024 and 2025 fits for backs agreed. The
# formation-based 2016-2024 charts favor usage even more (label weight 0.15-0.25).
DEPTH_CHART_FLOOR = 0.25
DEPTH_CHART_SHRINKAGE_GAMES = 1.5

# Same encoding as data_fetching/fetch_depth_charts.encode_depth.
ROLE_SCORES = {1: 1.0, 2: 0.5, 3: 0.2}
ROLE_SCORE_DEEP = 0.1

# The RB stats whose models read target share as a proxy for role. For these,
# receiving work may lift the projection but a thin share may not sink it.
RB_RUSHING_STATS = ("carries", "rushing_yards", "rushing_tds")

TEAM_ALIASES = {"LAR": "LA", "STL": "LA", "OAK": "LV", "SD": "LAC", "WSH": "WAS",
                "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA"}

DEF_STRENGTH_COLS = {"RB": "def_strength_vs_rb", "WR": "def_strength_vs_wr",
                     "TE": "def_strength_vs_te", "QB": "def_strength_vs_qb"}

DETAIL_FORMATS = {
    "QB": [("passing_attempts", "att", 0), ("passing_yards", "pass yd", 0), ("passing_tds", "pass TD", 1),
           ("interceptions", "INT", 1), ("rushing_yards", "rush yd", 0), ("rushing_tds", "rush TD", 1)],
    "RB": [("carries", "car", 1), ("rushing_yards", "rush yd", 0), ("rushing_tds", "rush TD", 1),
           ("receptions", "rec", 1), ("receiving_yards", "rec yd", 0), ("receiving_tds", "rec TD", 1)],
    "WR": [("receptions", "rec", 1), ("receiving_yards", "yd", 0), ("receiving_tds", "TD", 1)],
    "TE": [("receptions", "rec", 1), ("receiving_yards", "yd", 0), ("receiving_tds", "TD", 1)],
}

# What moves week to week, and what is still pinned to the prior season.
SEASON_FORM_NOTES = (
    f"team defensive strength and matchup advantage are still {CONTEXT_SEASON} "
    f"season averages (Layer 2 has not been run on {TARGET_SEASON})"
)


class PredictionError(Exception):
    pass


def fail(message: str) -> None:
    raise PredictionError(message)


def normalize_team(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    team = str(value).strip().upper()
    return TEAM_ALIASES.get(team, team)


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing {path}; run model/train.py first")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        fail(f"missing {label} at {path}")
    return pd.read_csv(path, low_memory=False)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator is None or numerator is None:
        return None
    if not np.isfinite(denominator) or denominator == 0:
        return None
    value = numerator / denominator
    return float(value) if np.isfinite(value) else None


def in_season_weight(games: int | None, shrinkage: float = SHRINKAGE_GAMES) -> float:
    """How far this season's sample may pull a feature off its prior-season value."""
    if not games:
        return 0.0
    return games / (games + shrinkage)


def blend(current: float | None, prior: float | None, weight: float) -> float | None:
    if current is None:
        return prior
    if prior is None:
        return current
    return weight * current + (1.0 - weight) * prior


def depth_chart_weight(games: int | None) -> float:
    """How much the depth chart label counts against the usage-implied role."""
    if not games:
        return 1.0
    fade = DEPTH_CHART_SHRINKAGE_GAMES / (DEPTH_CHART_SHRINKAGE_GAMES + games)
    return DEPTH_CHART_FLOOR + (1.0 - DEPTH_CHART_FLOOR) * fade


def shrink_ratio(ratio: float | None, weight: float) -> float | None:
    """Pull a this-season-over-last ratio toward 1.0, in log space so 0.5 and 2.0 are symmetric."""
    if ratio is None:
        return None
    return float(np.exp(weight * np.log(max(ratio, TREND_FLOOR))))


# ---------------------------------------------------------------------------
# Week selection
# ---------------------------------------------------------------------------

def regular_season(schedule: pd.DataFrame) -> pd.DataFrame:
    reg = schedule[schedule["game_type"] == "REG"].copy()
    reg["week"] = pd.to_numeric(reg["week"], errors="coerce")
    reg["gameday"] = pd.to_datetime(reg["gameday"], errors="coerce")
    return reg[reg["week"].notna()]


def resolve_week(schedule: pd.DataFrame, requested: int | None, today: date) -> tuple[int, str]:
    """Pick the week to project and explain the choice."""
    reg = regular_season(schedule)
    if reg.empty:
        fail(f"{SCHEDULE_PATH.name} has no regular-season games")

    weeks = sorted(int(w) for w in reg["week"].unique())

    if requested is not None:
        if requested not in weeks:
            fail(f"week {requested} is not in the {TARGET_SEASON} regular season "
                 f"(weeks {weeks[0]}-{weeks[-1]})")
        played = reg[(reg["week"] == requested) & reg["home_score"].notna()]
        if len(played) == len(reg[reg["week"] == requested]):
            return requested, f"week {requested} requested (already played - this is a backtest)"
        return requested, f"week {requested} requested"

    complete = reg.groupby("week")["home_score"].apply(lambda s: s.notna().all())
    last_gameday = reg.groupby("week")["gameday"].max()

    unplayed = [w for w in weeks if not bool(complete.get(w, False))]
    if not unplayed:
        return weeks[-1], f"{TARGET_SEASON} regular season is complete; showing week {weeks[-1]}"

    # A week whose last game has already kicked off but whose scores have not
    # reached the feed is not "upcoming" - skip to the first week still ahead.
    ahead = [w for w in unplayed
             if pd.notna(last_gameday.get(w)) and last_gameday[w].date() >= today]
    if ahead:
        week = ahead[0]
        kickoff = reg[reg["week"] == week]["gameday"].min()
        return week, f"next unplayed week as of {today:%Y-%m-%d} (week {week} opens {kickoff:%b %d})"

    week = unplayed[0]
    return week, (f"week {week} is the first without final scores, though its games are "
                  f"already past - the results feed may be behind")


# ---------------------------------------------------------------------------
# 2026 in-season form
# ---------------------------------------------------------------------------

class SeasonForm:
    """What the 2026 season has settled before the target week kicks off.

    Every lookup is restricted to weeks strictly before the target week, so a
    projection never sees the game it is projecting.
    """

    def __init__(self, target_week: int, schedule: pd.DataFrame):
        self.target_week = target_week
        self.weeks_available: list[int] = []

        box = read_optional_csv(BOX_SCORE_PATH)
        if not box.empty:
            box = box[box["season_type"] == "REG"].copy()
            box["week"] = pd.to_numeric(box["week"], errors="coerce")
            box = box[box["week"].notna() & (box["week"] < target_week)]
            if not box.empty:
                box["team"] = box["recent_team"].map(normalize_team)
                self.weeks_available = sorted(int(w) for w in box["week"].unique())
        self.box = box

        # The prior season is trimmed to the span 2026 has actually played, not to
        # the target week. Projecting week 7 off two played weeks would otherwise
        # divide two weeks of usage by six, reading a data gap as a collapse.
        prior = read_optional_csv(PRIOR_BOX_SCORE_PATH)
        if not prior.empty and self.weeks_available:
            prior = prior[prior["season_type"] == "REG"].copy()
            prior["week"] = pd.to_numeric(prior["week"], errors="coerce")
            prior = prior[prior["week"].notna() & (prior["week"] <= max(self.weeks_available))]
        elif not prior.empty:
            prior = prior.iloc[0:0]
        self.prior_box = prior

        # Last season in full, for the rushing baselines Layer 3 shrinks toward.
        prior_full = read_optional_csv(PRIOR_BOX_SCORE_PATH)
        if not prior_full.empty:
            prior_full = prior_full[prior_full["season_type"] == "REG"].copy()
            prior_full["week"] = pd.to_numeric(prior_full["week"], errors="coerce")
            prior_full = prior_full[prior_full["week"].notna()]
            prior_full["team"] = prior_full["recent_team"].map(normalize_team)
        self.prior_full_box = prior_full

        injuries = read_optional_csv(INJURY_PATH)
        if not injuries.empty:
            injuries = injuries[injuries["game_type"] == "REG"].copy()
            injuries["week"] = pd.to_numeric(injuries["week"], errors="coerce")
            injuries = injuries[injuries["week"].notna() & (injuries["week"] < target_week)
                                & injuries["report_status"].isin(DNP_STATUSES)]
        self.injuries = injuries

        self.depth_chart, self.depth_week = self._latest_depth_chart()
        self.depth_snapshot = None
        if "snapshot_dt" in self.depth_chart.columns:
            stamp = pd.to_datetime(self.depth_chart["snapshot_dt"], errors="coerce").max()
            self.depth_snapshot = None if pd.isna(stamp) else stamp
        self.oline = self._oline_ranks(OLINE_PATH, lambda week: week < target_week)
        self.prior_oline = self._oline_ranks(PRIOR_OLINE_PATH, lambda week: week.notna())
        self.bye_weeks = self._bye_weeks(schedule)

    # -- loaders ------------------------------------------------------------

    def _latest_depth_chart(self) -> tuple[pd.DataFrame, int | None]:
        """The most recent chart published at or before the target week."""
        chart = read_optional_csv(DEPTH_CHART_PATH)
        if chart.empty:
            return chart, None
        chart["week"] = pd.to_numeric(chart["week"], errors="coerce")
        eligible = chart[chart["week"].notna() & (chart["week"] <= self.target_week)]
        if eligible.empty:
            # Projecting before any chart exists; fall back to the earliest one.
            eligible = chart[chart["week"] == chart["week"].min()]
        if eligible.empty:
            return eligible, None
        week = int(eligible["week"].max())
        return eligible[eligible["week"] == week], week

    @staticmethod
    def _oline_ranks(path: Path, keep) -> pd.DataFrame:
        """Weekly o-line ranks. Each is one game's rank, so callers average them."""
        oline = read_optional_csv(path)
        if oline.empty:
            return oline
        oline["week"] = pd.to_numeric(oline["week"], errors="coerce")
        oline = oline[oline["week"].notna() & keep(oline["week"])].copy()
        oline["team"] = oline["team"].map(normalize_team)
        return oline

    @staticmethod
    def _bye_weeks(schedule: pd.DataFrame) -> dict[str, int]:
        """A team's bye is the one regular-season week it has no game."""
        reg = regular_season(schedule)
        played_weeks: dict[str, set[int]] = {}
        for column in ("home_team", "away_team"):
            for team, week in zip(reg[column].map(normalize_team), reg["week"]):
                if team is not None and pd.notna(week):
                    played_weeks.setdefault(team, set()).add(int(week))

        byes = {}
        for team, weeks in played_weeks.items():
            missing = set(range(min(weeks), max(weeks) + 1)) - weeks
            if len(missing) == 1:
                byes[team] = next(iter(missing))
        return byes

    # -- player lookups -----------------------------------------------------

    def has_data(self) -> bool:
        return bool(self.weeks_available)

    def _player_rows(self, frame: pd.DataFrame, player_id, name: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        rows = frame[frame["player_id"] == player_id] if player_id is not None else frame.iloc[0:0]
        if rows.empty and "player_display_name" in frame.columns:
            rows = frame[frame["player_display_name"].str.casefold() == name.casefold()]
        return rows

    def games_played(self, player_id, name: str) -> int | None:
        if not self.has_data():
            return None
        return int(len(self._player_rows(self.box, player_id, name)))

    def injury_history(self, player_id) -> tuple[float, float] | None:
        """(prev_injuries_this_szn, weeks_since_injury) from 2026 injury reports."""
        if self.injuries.empty or player_id is None:
            return None
        rows = self.injuries[self.injuries["gsis_id"] == player_id]
        if rows.empty:
            return 0.0, NO_INJURY_WEEKS
        last = float(rows["week"].max())
        return float(len(rows)), float(self.target_week) - last

    def target_share(self, player_id, name: str, team: str) -> float | None:
        """Player targets over team targets, both cumulative through last week."""
        if not self.has_data():
            return None
        rows = self._player_rows(self.box, player_id, name)
        if rows.empty:
            return None
        team_rows = self.box[self.box["team"] == team]
        return safe_ratio(float(rows["targets"].fillna(0.0).sum()),
                          float(team_rows["targets"].fillna(0.0).sum()))

    def usage_trends(self, player_id, name: str) -> tuple[float | None, float | None]:
        """This season's usage and target share against the same span last season.

        Mirrors Layer 3's usage_trend / target_share_trend, with both sides cut to
        the same number of weeks so the ratio stays comparable.
        """
        if not self.has_data() or self.prior_box.empty:
            return None, None

        now = self._player_rows(self.box, player_id, name)
        before = self._player_rows(self.prior_box, player_id, name)
        if now.empty or before.empty:
            return None, None

        def usage(rows: pd.DataFrame) -> float:
            return float(rows["targets"].fillna(0.0).sum() + rows["carries"].fillna(0.0).sum())

        def avg_share(rows: pd.DataFrame) -> float | None:
            shares = rows["target_share"].fillna(0.0)
            return float(shares.mean()) if len(shares) else None

        usage_trend = safe_ratio(usage(now), usage(before))
        share_now, share_before = avg_share(now), avg_share(before)
        share_trend = (safe_ratio(share_now, share_before)
                       if share_now is not None and share_before is not None else None)
        return usage_trend, share_trend

    def rushing_features(self, player_id, name: str, team: str) -> tuple[dict[str, float], str | None]:
        """Layer 3's rushing features as of the target week.

        This season's carries and yards through last week, shrunk toward the
        player's full previous season with the same constants Layer 3 uses, and
        the average of the last few games played across both seasons.
        """
        now = self._player_rows(self.box, player_id, name) if self.has_data() else pd.DataFrame()
        before = self._player_rows(self.prior_full_box, player_id, name)
        if now.empty and before.empty:
            return {}, None

        def total(rows: pd.DataFrame, column: str) -> float:
            return float(rows[column].fillna(0.0).sum()) if not rows.empty else 0.0

        games = len(now)
        carries, yards = total(now, "carries"), total(now, "rushing_yards")
        team_carries = total(self.box[self.box["team"] == team], "carries") if games else 0.0

        last_games = len(before)
        last_carries, last_yards = total(before, "carries"), total(before, "rushing_yards")
        last_team_carries = None
        if last_games:
            last_team = before.sort_values("week")["team"].iloc[-1]
            last_team_carries = total(self.prior_full_box[self.prior_full_box["team"] == last_team], "carries")

        def shrink(current: float | None, prior: float | None, k: float) -> float | None:
            return blend(current if games else None, prior, games / (games + k) if games else 0.0)

        carries_pg = shrink(safe_ratio(carries, games), safe_ratio(last_carries, last_games),
                            RUSH_VOLUME_SHRINKAGE_GAMES)
        yards_pg = shrink(safe_ratio(yards, games), safe_ratio(last_yards, last_games),
                          RUSH_YARDS_SHRINKAGE_GAMES)
        share = shrink(safe_ratio(carries, team_carries),
                       safe_ratio(last_carries, last_team_carries) if last_team_carries else None,
                       RUSH_VOLUME_SHRINKAGE_GAMES)

        values: dict[str, float] = {}
        for column, value in (("carries_per_game", carries_pg), ("rush_yds_per_game", yards_pg),
                              ("carry_share_of_team", share)):
            if value is not None:
                values[column] = value
        if carries_pg and yards_pg is not None and carries + last_carries >= MIN_CARRIES_FOR_YPC:
            values["yards_per_carry"] = yards_pg / carries_pg

        recent = pd.concat([before.assign(_order=0), now.assign(_order=1)], ignore_index=True)
        recent = recent.sort_values(["_order", "week"]).tail(RECENT_GAMES)
        if not recent.empty:
            values["carries_last3"] = float(recent["carries"].fillna(0.0).mean())

        detail = ", ".join(f"{column} {values[column]:.{3 if 'share' in column else 1}f}"
                           for column in ("carries_per_game", "rush_yds_per_game", "yards_per_carry",
                                          "carry_share_of_team", "carries_last3") if column in values)
        return values, f"{detail} ({games} {TARGET_SEASON} game(s) + {last_games} {CONTEXT_SEASON})"

    def receiving_features(self, player_id, name: str, team: str) -> tuple[dict[str, float], str | None]:
        """Layer 3's receiving volume, efficiency and team pass volume as of the target week.

        Same construction as rushing_features: this season through last week,
        shrunk toward the player's full previous season (the team's, for pass
        attempts) with the constants Layer 3 was fit with.
        """
        values: dict[str, float] = {}
        details = []

        # Team pass attempts belong to the team, so they are set for every player.
        team_now = self.box[self.box["team"] == team] if self.has_data() else pd.DataFrame()
        team_before = (self.prior_full_box[self.prior_full_box["team"] == team]
                       if not self.prior_full_box.empty else pd.DataFrame())
        team_games = int(team_now["week"].nunique()) if not team_now.empty else 0
        last_team_games = int(team_before["week"].nunique()) if not team_before.empty else 0
        attempts_pg = blend(
            safe_ratio(float(team_now["attempts"].fillna(0.0).sum()), team_games) if team_games else None,
            safe_ratio(float(team_before["attempts"].fillna(0.0).sum()), last_team_games) if last_team_games else None,
            in_season_weight(team_games, TEAM_PASS_SHRINKAGE_GAMES))
        if attempts_pg is not None:
            values["team_pass_att_per_game"] = attempts_pg
            details.append(f"team pass att/g {attempts_pg:.1f}")

        now = self._player_rows(self.box, player_id, name) if self.has_data() else pd.DataFrame()
        before = self._player_rows(self.prior_full_box, player_id, name)
        if now.empty and before.empty:
            return values, ", ".join(details) or None

        def total(rows: pd.DataFrame, column: str) -> float:
            return float(rows[column].fillna(0.0).sum()) if not rows.empty else 0.0

        games, last_games = len(now), len(before)

        def shrink(current: float | None, prior: float | None, k: float) -> float | None:
            return blend(current if games else None, prior, games / (games + k) if games else 0.0)

        targets, last_targets = total(now, "targets"), total(before, "targets")
        team_air = total(team_now, "receiving_air_yards")
        last_team_air = None
        if last_games:
            last_team = before.sort_values("week")["team"].iloc[-1]
            last_team_air = total(self.prior_full_box[self.prior_full_box["team"] == last_team],
                                  "receiving_air_yards")

        for column, stat in (("targets_per_game", "targets"), ("rec_yds_per_game", "receiving_yards")):
            value = shrink(safe_ratio(total(now, stat), games), safe_ratio(total(before, stat), last_games),
                           REC_VOLUME_SHRINKAGE_GAMES)
            if value is not None:
                values[column] = value
        air_share = shrink(safe_ratio(total(now, "receiving_air_yards"), team_air),
                           safe_ratio(total(before, "receiving_air_yards"), last_team_air) if last_team_air else None,
                           REC_VOLUME_SHRINKAGE_GAMES)
        if air_share is not None:
            values["player_air_yards_share"] = air_share

        if targets + last_targets >= MIN_TARGETS_FOR_RATE:
            for column, stat in (("yds_per_target", "receiving_yards"), ("catch_rate", "receptions"),
                                 ("air_yds_per_target", "receiving_air_yards"), ("td_per_target", "receiving_tds")):
                value = shrink(safe_ratio(total(now, stat), targets), safe_ratio(total(before, stat), last_targets),
                               REC_RATE_SHRINKAGE_GAMES[column])
                if value is not None:
                    values[column] = value

        formats = (("targets_per_game", "tgt/g", ".1f"), ("rec_yds_per_game", "yd/g", ".1f"),
                   ("yds_per_target", "yd/tgt", ".2f"), ("catch_rate", "catch", ".0%"),
                   ("air_yds_per_target", "aDOT", ".1f"), ("td_per_target", "TD/tgt", ".3f"),
                   ("player_air_yards_share", "air share", ".1%"))
        details = [f"{label} {values[column]:{spec}}" for column, label, spec in formats
                   if column in values] + details
        return values, f"{', '.join(details)} ({games} {TARGET_SEASON} game(s) + {last_games} {CONTEXT_SEASON})"

    def depth_score(self, player_id, name: str, position: str) -> tuple[float, str]:
        if self.depth_chart.empty:
            return DEPTH_SCORE_UNKNOWN, f"no {DEPTH_CHART_PATH.name}"

        rows = self._player_rows_by_name(self.depth_chart, player_id, name)
        if rows.empty:
            return DEPTH_SCORE_UNKNOWN, "not on depth chart"

        exact = rows[rows["position"].astype("string").str.upper() == position]
        chosen = (exact if not exact.empty else rows).sort_values("depth_chart_rank")
        rank = chosen.iloc[0]["depth_chart_rank"]
        score = float(chosen.iloc[0]["depth_chart_position"])
        source = f"{position}{int(rank)} on week {self.depth_week} chart"
        if self.depth_snapshot is not None:
            source += f", snapshot {self.depth_snapshot:%b %d %H:%M} UTC"
        return score, source

    def usage_role(self, player_id, name: str, team: str, position: str) -> tuple[float, str] | None:
        """The depth tier this season's usage implies, through last week.

        Rank in opportunities per game played among the team's players at the
        position: targets for receivers, carries plus targets for backs. Ties go
        to the player, and only games for his current team count.
        """
        if not self.has_data() or "position" not in self.box.columns:
            return None
        rows = self._player_rows(self.box, player_id, name)
        rows = rows[rows["team"] == team]
        if rows.empty:
            return None

        group = self.box[(self.box["team"] == team)
                         & (self.box["position"].astype("string").str.upper() == position)]
        group = pd.concat([group, rows]).drop_duplicates(subset=["player_id", "week"])
        opportunities = group["targets"].fillna(0.0)
        label = "tgt"
        if position == "RB":
            opportunities = opportunities + group["carries"].fillna(0.0)
            label = "car+tgt"
        per_game = opportunities.groupby(group["player_id"]).mean()

        mine = float(per_game[rows["player_id"].iloc[0]])
        rank = int((per_game > mine).sum()) + 1
        score = ROLE_SCORES.get(rank, ROLE_SCORE_DEEP)
        return score, (f"{position}{rank} by {TARGET_SEASON} usage ({mine:.1f} {label}/g, "
                       f"of {len(per_game)} {team} {position}s)")

    @staticmethod
    def _player_rows_by_name(frame: pd.DataFrame, player_id, name: str) -> pd.DataFrame:
        rows = frame[frame["player_id"] == player_id] if player_id is not None else frame.iloc[0:0]
        if rows.empty:
            rows = frame[frame["player_name"].str.casefold() == name.casefold()]
        return rows

    def oline_rank(self, team: str) -> tuple[float, str] | None:
        """This season's average rank, shrunk toward last season's average.

        One week's rank can swing from top five to bottom five on a single game,
        so two weeks of it should not replace a full season of it.
        """
        if self.oline.empty:
            return None
        now = self.oline[self.oline["team"] == team]["oline_rank"]
        if now.empty:
            return None
        before = (self.prior_oline[self.prior_oline["team"] == team]["oline_rank"]
                  if not self.prior_oline.empty else pd.Series(dtype="float64"))
        prior = float(before.mean()) if len(before) else None
        rank = blend(float(now.mean()), prior, in_season_weight(len(now)))
        detail = f"{now.mean():.1f} over {len(now)} wk"
        if prior is not None:
            detail += f", blended with {CONTEXT_SEASON} avg {prior:.1f} -> {rank:.1f}"
        return rank, detail

    def bye_passed(self, team: str) -> float | None:
        bye = self.bye_weeks.get(team)
        return None if bye is None else float(self.target_week > bye)

    def summary(self) -> str:
        if not self.has_data():
            return f"no {TARGET_SEASON} games played yet; all form carried from {CONTEXT_SEASON}"
        span = f"{self.weeks_available[0]}-{self.weeks_available[-1]}"
        return f"{TARGET_SEASON} weeks {span}"


# ---------------------------------------------------------------------------
# Baseline context (prior season) and feature assembly
# ---------------------------------------------------------------------------

def find_player(bio: pd.DataFrame, name: str) -> pd.Series:
    exact = bio[bio["player_name"] == name]
    if exact.empty:
        exact = bio[bio["player_name"].str.casefold() == name.casefold()]
    if exact.empty:
        exact = bio[bio["player_name"].str.contains(name, case=False, regex=False, na=False)]
    if exact.empty:
        fail(f"Player '{name}' not found in {TARGET_SEASON} rosters")

    ranked = exact.copy()
    ranked["_rank"] = ranked["position"].isin(DEF_STRENGTH_COLS).astype(int)
    # Prefer the most recent roster snapshot, so an in-season move is picked up.
    sort_cols = ["_rank"] + (["week"] if "week" in ranked.columns else [])
    ranked = ranked.sort_values(sort_cols, ascending=False)
    return ranked.iloc[0]


def find_game(schedule: pd.DataFrame, team: str, week: int) -> tuple[pd.Series, bool]:
    games = regular_season(schedule)
    games = games[games["week"] == week].copy()
    games["home_team"] = games["home_team"].map(normalize_team)
    games["away_team"] = games["away_team"].map(normalize_team)

    home = games[games["home_team"] == team]
    if not home.empty:
        return home.iloc[0], True

    away = games[games["away_team"] == team]
    if not away.empty:
        return away.iloc[0], False

    fail(f"{team} has no week {week} game in {SCHEDULE_PATH.name} - it is their bye week")


def lookup_row(frame: pd.DataFrame, team_col: str, team: str) -> pd.Series | None:
    if frame.empty or team_col not in frame.columns:
        return None
    match = frame[frame[team_col].map(normalize_team) == team]
    return None if match.empty else match.iloc[0]


def player_context_row(training: pd.DataFrame, player_id: str, name: str, position: str,
                       feature_cols: list[str]) -> tuple[pd.Series, str]:
    rows = training[training["player_id"] == player_id]
    if rows.empty:
        rows = training[training["player_name"].str.casefold() == name.casefold()]

    if not rows.empty:
        rows = rows.sort_values("week")
        played = rows[rows["dnp_flag"] != 1] if "dnp_flag" in rows.columns else rows
        source = f"{CONTEXT_SEASON} game log"
        if played.empty:
            played = rows
            source = f"{CONTEXT_SEASON} game log (no games played)"
        elif len(played) < len(rows):
            source = f"{CONTEXT_SEASON} game log, {len(rows) - len(played)} DNP week(s) skipped"
        return played[feature_cols].ffill().iloc[-1], source

    peers = training[training["bio_position"] == position]
    if peers.empty:
        return pd.Series({c: np.nan for c in feature_cols}, dtype="float64"), "no prior data"
    return peers[feature_cols].median(numeric_only=True), f"{CONTEXT_SEASON} {position} median"


def prior_season_target_share(training: pd.DataFrame, player_id, name: str) -> float | None:
    if "targets" not in training.columns or "team" not in training.columns:
        return None

    rows = training[training["player_id"] == player_id]
    if rows.empty:
        rows = training[training["player_name"].str.casefold() == name.casefold()]
    if rows.empty:
        return None

    player_targets = float(rows["targets"].fillna(0.0).sum())
    player_team = rows.sort_values("week")["team"].iloc[-1]
    team_targets = float(training[training["team"] == player_team]["targets"].fillna(0.0).sum())
    return safe_ratio(player_targets, team_targets)


def peer_target_share(training: pd.DataFrame, position: str, depth_score: float) -> float | None:
    """Median prior-season target share of players at the same position and depth tier."""
    peers = training[(training["bio_position"] == position)
                     & (training["depth_chart_position"] == depth_score)]
    if "dnp_flag" in peers.columns:
        peers = peers[peers["dnp_flag"] != 1]
    shares = peers["player_target_share_of_team"].dropna()
    return float(shares.median()) if len(shares) else None


def draft_reference_values(training: pd.DataFrame, position: str) -> list[float]:
    """Deciles of the position's draft slots, one value per player.

    Undrafted players count as 0, the value train.py's zero-fill gives them, so
    this is the distribution the models actually saw.
    """
    peers = training[training["bio_position"] == position]
    if peers.empty or "bio_draft_number" not in peers.columns:
        return []
    slots = peers.groupby("player_id")["bio_draft_number"].last().fillna(0.0)
    return [float(v) for v in np.quantile(slots, DRAFT_REFERENCE_QUANTILES, method="lower")]


def draft_weight(bio: pd.Series) -> tuple[float, int | None]:
    """(weight on the player's own draft slot, NFL seasons before this one)."""
    first = next((bio.get(c) for c in ("rookie_year", "entry_year") if pd.notna(bio.get(c))), None)
    if first is not None:
        seasons = max(0, TARGET_SEASON - int(first))
    elif pd.notna(bio.get("years_exp")):
        seasons = max(0, int(bio.get("years_exp")))
    else:
        return 1.0, None
    return DRAFT_WEIGHT_DECAY ** seasons, seasons


def build_overrides(game: pd.Series, is_home: bool, bio: pd.Series, team: str, position: str,
                    week: int, oc: pd.Series | None, sos: pd.Series | None,
                    team_def: pd.Series | None, opp_def: pd.Series | None,
                    league_avg: float | None, depth_score: float,
                    target_share: float | None, form: SeasonForm,
                    notes: dict[str, str]) -> dict[str, float]:
    spread = game.get("spread_line")
    total = game.get("total_line")
    spread = DEFAULT_SPREAD_LINE if pd.isna(spread) else float(spread)
    total = DEFAULT_TOTAL_LINE if pd.isna(total) else float(total)

    rest = game.get("home_rest") if is_home else game.get("away_rest")

    # Weather is only published once a game is close; future weeks get defaults.
    temp = game.get("temp")
    wind = game.get("wind")
    temp = DEFAULT_WEATHER_TEMP if pd.isna(temp) else float(temp)
    wind = DEFAULT_WEATHER_WIND if pd.isna(wind) else float(wind)

    values: dict[str, float] = {
        "week": float(week),
        "weeks_into_season": float(week),
        "is_home": 1.0 if is_home else 0.0,
        "spread_line": spread,
        "total_line": total,
        "implied_home_total": (total - spread) / 2,
        "implied_away_total": (total + spread) / 2,
        "weather_temp": temp,
        "weather_wind": wind,
        "sched_temp": temp,
        "sched_wind": wind,
        "rest_days": float(rest) if pd.notna(rest) else 7.0,
        "depth_chart_position": float(depth_score),
    }

    player_id = bio.get("player_id")
    player_name = bio.get("player_name", "")

    games_played = form.games_played(player_id, player_name)
    values["games_played_this_szn"] = float(games_played) if games_played is not None else 0.0

    bye_passed = form.bye_passed(team)
    if bye_passed is not None:
        values["bye_week_passed"] = bye_passed

    injuries = form.injury_history(player_id)
    if injuries is not None:
        values["prev_injuries_this_szn"], values["weeks_since_injury"] = injuries
    else:
        values["prev_injuries_this_szn"] = 0.0

    weight = in_season_weight(games_played, TREND_SHRINKAGE_GAMES)
    usage_trend, share_trend = form.usage_trends(player_id, player_name)
    trend_notes = []
    for column, raw in (("usage_trend", usage_trend), ("target_share_trend", share_trend)):
        if raw is not None:
            values[column] = shrink_ratio(raw, weight)
            trend_notes.append(f"{column} {raw:.2f} -> {values[column]:.2f}")
    if trend_notes:
        notes["trends"] = ", ".join(trend_notes) + f" ({TARGET_SEASON} weight {weight:.0%})"

    rushing, rushing_detail = form.rushing_features(player_id, player_name, team)
    values.update(rushing)
    if rushing_detail:
        notes["rushing"] = rushing_detail

    receiving, receiving_detail = form.receiving_features(player_id, player_name, team)
    values.update(receiving)
    if receiving_detail:
        notes["receiving"] = receiving_detail

    oline = form.oline_rank(team)
    if oline is not None:
        values["oline_rank"], notes["oline"] = oline

    if target_share is not None:
        values["player_target_share_of_team"] = float(target_share)

    for column in ("away_rest", "home_rest", "away_moneyline", "home_moneyline",
                   "away_spread_odds", "home_spread_odds", "under_odds", "over_odds", "div_game"):
        value = game.get(column)
        if pd.notna(value):
            values[column] = float(value)

    for source, destination in (("age", "bio_age"), ("height", "bio_height"), ("weight", "bio_weight"),
                                ("years_exp", "bio_years_exp"), ("entry_year", "bio_entry_year"),
                                ("rookie_year", "bio_rookie_year")):
        value = bio.get(source)
        if pd.notna(value):
            values[destination] = float(value)

    draft_number = bio.get("draft_number")
    if pd.notna(draft_number):
        values["bio_draft_number"] = float(draft_number)

    if oc is not None:
        # Once Layer 1 includes a season whose coordinator file carries these
        # columns, the merge with team_aggs suffixes them _x (coordinator) and _y
        # (team_aggs); the models train on _y, the only one populated historically.
        for column in ("team_target_share_to_wr", "team_target_share_to_te",
                       "team_target_share_to_rb", "team_rush_share_to_rb"):
            value = oc.get(column)
            if pd.notna(value):
                for destination in (column, f"{column}_x", f"{column}_y"):
                    values[destination] = float(value)

    if sos is not None:
        for source, destinations in (("preseason_sos", ("preseason_sos_x", "preseason_sos_y")),
                                     ("adjusted_sos", ("adjusted_sos_x", "adjusted_sos_y"))):
            value = sos.get(source)
            if pd.notna(value):
                for destination in destinations:
                    values[destination] = float(value)

    if team_def is not None:
        for column in DEF_STRENGTH_COLS.values():
            value = team_def.get(column)
            if pd.notna(value):
                values[column] = float(value)

    if opp_def is not None and position in DEF_STRENGTH_COLS:
        opp_strength = opp_def.get(DEF_STRENGTH_COLS[position])
        if pd.notna(opp_strength):
            values["opp_def_strength_for_position"] = float(opp_strength)
            if league_avg is not None and not pd.isna(league_avg):
                values["matchup_advantage_score"] = float(opp_strength) - float(league_avg)

    return values


def assemble_vector(feature_cols: list[str], context: pd.Series, overrides: dict[str, float],
                    scaler_mean: list[float]) -> tuple[np.ndarray, list[str]]:
    raw = []
    defaulted = []
    for index, column in enumerate(feature_cols):
        if column in overrides and pd.notna(overrides[column]):
            raw.append(float(overrides[column]))
            continue
        value = context.get(column, np.nan)
        if pd.isna(value):
            raw.append(float(scaler_mean[index]))
            defaulted.append(column)
        else:
            raw.append(float(value))
    return np.asarray(raw, dtype="float64"), defaulted


def expand_vector(vector: np.ndarray, feature_cols: list[str],
                  choices: dict[str, list[tuple[float, float]]]) -> list[tuple[float, np.ndarray]]:
    """Every combination of the weighted alternative values, with its joint weight."""
    columns = [c for c in choices if c in feature_cols]
    mixes = []
    for combo in itertools.product(*(choices[c] for c in columns)):
        variant = vector.copy()
        weight = 1.0
        for column, (option_weight, value) in zip(columns, combo):
            variant[feature_cols.index(column)] = value
            weight *= option_weight
        if weight > 0:
            mixes.append((weight, variant))
    return mixes


def mixture_predict(model, mixes: list[tuple[float, np.ndarray]], normalize) -> float:
    """Weighted average of one model's predictions over alternative input vectors."""
    weights = np.asarray([w for w, _ in mixes])
    batch = np.vstack([normalize(v) for _, v in mixes])
    return float(weights @ model.predict(batch) / weights.sum())


def refresh_depth_chart() -> str | None:
    """Re-fetch this season's depth chart if the saved copy is stale.

    Safe for backtests too: the fetcher gives each week the last snapshot before
    its first kickoff, so a fresh download changes only the upcoming week.
    """
    if not REFRESH_DEPTH_CHART:
        return None
    if DEPTH_CHART_PATH.exists():
        age_hours = (time.time() - DEPTH_CHART_PATH.stat().st_mtime) / 3600
        if age_hours < DEPTH_REFRESH_HOURS:
            return None
    try:
        sys.path.insert(0, str(PROJECT_DIR / "data_fetching"))
        from fetch_depth_charts import save_season
        return save_season(TARGET_SEASON)
    except Exception as exc:
        return f"failed ({type(exc).__name__}: {exc}); using the saved copy"


def load_models(position_meta: dict) -> dict:
    models = {}
    for stat, info in position_meta.items():
        path = MODEL_DIR / info["model_path"]
        if not path.exists():
            fail(f"missing model {path.name}; run model/train.py first")
        with path.open("rb") as fh:
            models[stat] = pickle.load(fh)
    return models


def format_detail(position: str, predictions: dict[str, float]) -> str:
    parts = []
    for stat, label, digits in DETAIL_FORMATS.get(position, []):
        if stat in predictions:
            parts.append(f"{predictions[stat]:.{digits}f} {label}")
    return ", ".join(parts)


def predict(name: str, requested_week: int | None = None, today: date | None = None) -> None:
    today = today or date.today()

    metadata = load_json(METADATA_PATH)
    bio = load_csv(BIO_PATH, f"{TARGET_SEASON} bio data")
    schedule = load_csv(SCHEDULE_PATH, f"{TARGET_SEASON} schedule")

    week, week_reason = resolve_week(schedule, requested_week, today)

    player = find_player(bio, name)
    position = str(player.get("position", "")).upper()
    player_name = player["player_name"]

    if position not in metadata["positions"]:
        fail(f"{player_name} plays {position or 'an unknown position'}; "
             f"models exist only for {', '.join(metadata['positions'])}")

    team = normalize_team(player.get("team"))
    if team is None:
        fail(f"{player_name} has no {TARGET_SEASON} team assigned")

    game, is_home = find_game(schedule, team, week)
    opponent = normalize_team(game["away_team"] if is_home else game["home_team"])

    position_meta = metadata["positions"][position]
    reference = next(iter(position_meta.values()))
    feature_cols = reference["features"]
    scaler_mean = reference["scaler_mean"]
    scaler_scale = reference["scaler_scale"]

    training = pd.read_parquet(
        TRAINING_PATH,
        columns=sorted(set(feature_cols) | {"season", "week", "player_id", "player_name",
                                            "bio_position", "dnp_flag", "targets", "team"}),
        filters=[("season", "==", float(CONTEXT_SEASON))],
    )
    context, context_source = player_context_row(training, player.get("player_id"), player_name,
                                                 position, feature_cols)

    refreshed = refresh_depth_chart()
    form = SeasonForm(week, schedule)

    oc = lookup_row(load_csv(OC_PATH, f"{TARGET_SEASON} coordinator data"), "team", team)
    sos = lookup_row(load_csv(SOS_PATH, f"{TARGET_SEASON} strength of schedule"), "team", team)

    team_def = opp_def = None
    league_avg = None
    if TEAM_AGGS_PATH.exists():
        aggs = pd.read_parquet(TEAM_AGGS_PATH)
        aggs = aggs[aggs["season"] == float(CONTEXT_SEASON)]
        team_def = lookup_row(aggs, "team", team)
        opp_def = lookup_row(aggs, "team", opponent)
        if position in DEF_STRENGTH_COLS and not aggs.empty:
            league_avg = aggs[DEF_STRENGTH_COLS[position]].mean()

    depth_score, depth_source = form.depth_score(player.get("player_id"), player_name, position)

    # Last season's full-year share is the baseline; 2026 games pull it toward
    # this season's share only as fast as the sample size justifies. A player
    # with no prior season (a rookie) starts from the median of his depth tier.
    games = form.games_played(player.get("player_id"), player_name) or 0
    weight = in_season_weight(games)
    prior_share = prior_season_target_share(training, player.get("player_id"), player_name)
    prior_basis = f"{CONTEXT_SEASON} season"
    if prior_share is None:
        prior_share = peer_target_share(training, position, depth_score)
        prior_basis = f"{CONTEXT_SEASON} {position} depth-tier median"
    season_share = form.target_share(player.get("player_id"), player_name, team)
    target_share = blend(season_share, prior_share, weight)
    if season_share is None:
        share_basis = prior_basis
    else:
        prior_text = "n/a" if prior_share is None else f"{prior_share:.1%}"
        share_basis = (f"{TARGET_SEASON} to date {season_share:.1%} x{weight:.0%} + "
                       f"{prior_basis} {prior_text} x{1 - weight:.0%}")

    notes: dict[str, str] = {}
    overrides = build_overrides(game, is_home, player, team, position, week,
                                oc, sos, team_def, opp_def, league_avg, depth_score,
                                target_share, form, notes)
    raw_vector, defaulted = assemble_vector(feature_cols, context, overrides, scaler_mean)

    # The RB rushing models read target share as a role signal, so a pure
    # between-the-tackles back would lose carries for not catching passes. They
    # also see the typical share of his depth tier and no falling share trend,
    # and keep whichever prediction is higher: tree models are not monotone in
    # share, so only the max guarantees receiving work can help but never hurt.
    # The receiving models still see the real share.
    rushing_vector = None
    if position == "RB":
        rushing_vector = raw_vector.copy()
        share_index = feature_cols.index("player_target_share_of_team")
        trend_index = feature_cols.index("target_share_trend")
        share_floor = peer_target_share(training, position, depth_score)
        floored = []
        if share_floor is not None and rushing_vector[share_index] < share_floor:
            floored.append(f"share {rushing_vector[share_index]:.1%} -> depth-tier median {share_floor:.1%}")
            rushing_vector[share_index] = share_floor
        if rushing_vector[trend_index] < 1.0:
            floored.append(f"share trend {rushing_vector[trend_index]:.2f} -> 1.00")
            rushing_vector[trend_index] = 1.0
        notes["rb_floor"] = ("floored " + ", ".join(floored) if floored
                             else "share and trend at or above neutral, no floor needed")

    # Draft slot and depth chart label are re-weighted by averaging the models
    # over alternative values of each, not by editing the value itself: the
    # trees split these features into steps, so a value between two tiers would
    # just land on one side of a split.
    choices: dict[str, list[tuple[float, float]]] = {}
    if position in ROLE_BLEND_POSITIONS:
        chart_weight = depth_chart_weight(games)
        role = form.usage_role(player.get("player_id"), player_name, team, position)
        if role is not None and chart_weight < 1.0:
            role_score, notes["role"] = role
            choices["depth_chart_position"] = [(chart_weight, depth_score), (1.0 - chart_weight, role_score)]
            notes["role"] += f"; chart label weight {chart_weight:.0%}, usage role {1 - chart_weight:.0%}"
        else:
            notes["role"] = f"no {TARGET_SEASON} usage yet; chart label weight 100%"

        own_weight, seasons = draft_weight(player)
        references = draft_reference_values(training, position)
        if "bio_draft_number" in feature_cols and references and own_weight < 1.0:
            own_slot = raw_vector[feature_cols.index("bio_draft_number")]
            choices["bio_draft_number"] = ([(own_weight, own_slot)]
                                           + [((1.0 - own_weight) / len(references), v) for v in references])
        slot = player.get("draft_number")
        slot_text = f"pick #{int(slot)}" if pd.notna(slot) else "undrafted"
        season_text = ("rookie" if seasons == 0
                       else f"{seasons} prior NFL season(s)" if seasons else "tenure unknown")
        notes["draft"] = f"{slot_text}, {season_text}: own slot weight {own_weight:.0%}"
        if own_weight < 1.0:
            notes["draft"] += f", rest averaged over {CONTEXT_SEASON} {position} draft slots"

    def normalize(vector: np.ndarray) -> np.ndarray:
        return ((vector - np.asarray(scaler_mean)) / np.asarray(scaler_scale)).reshape(1, -1)

    models = load_models(position_meta)
    predictions = {}
    lifted = []
    base_mix = expand_vector(raw_vector, feature_cols, choices)
    rushing_mix = expand_vector(rushing_vector, feature_cols, choices) if rushing_vector is not None else None
    for stat, model in models.items():
        value = mixture_predict(model, base_mix, normalize)
        if rushing_mix is not None and stat in RB_RUSHING_STATS:
            floored_value = mixture_predict(model, rushing_mix, normalize)
            if floored_value > value:
                lifted.append(f"{stat} +{floored_value - value:.1f}")
                value = floored_value
        predictions[stat] = max(0.0, value)
    if rushing_vector is not None and lifted:
        notes["rb_floor"] += f"; floor lifted {', '.join(lifted)}"
    elif rushing_vector is not None:
        notes["rb_floor"] += "; floor did not change the projection"

    weights = metadata["ppr_weights"]
    points = max(0.0, sum(predictions.get(stat, 0.0) * weight for stat, weight in weights.items()))

    location = "vs" if is_home else "at"
    kickoff = game.get("gameday")
    kickoff_text = f", {pd.to_datetime(kickoff):%b %d}" if pd.notna(kickoff) else ""

    print(f"{player_name} ({position}, {team} {location} {opponent}, Week {week}{kickoff_text})")
    print(f"Projected: {points:.1f} PPR")
    detail = format_detail(position, predictions)
    if detail:
        print(f"({detail})")
    share_text = "n/a" if target_share is None else f"{target_share:.1%}"
    print(f"[week: {week_reason}]")
    print(f"[form: {form.summary()}; {games} game(s) played, {TARGET_SEASON} weight {weight:.0%}; "
          f"depth: {depth_source} ({depth_score:.2f})]")
    print(f"[target share: {share_text} ({share_basis})]")
    for key, label in (("role", "role"), ("draft", "draft"), ("trends", "trends"),
                       ("rushing", "rushing form"), ("receiving", "receiving form"),
                       ("oline", "o-line rank"), ("rb_floor", "RB floor")):
        if key in notes:
            print(f"[{label}: {notes[key]}]")
    print(f"[context: {context_source}; {len(defaulted)} of {len(feature_cols)} features unavailable]")
    if refreshed:
        print(f"[depth chart refresh: {refreshed}]")
    print(f"[note: {SEASON_FORM_NOTES}]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Project a player's PPR line for a {TARGET_SEASON} week.",
        epilog="With no --week, the next unplayed week of the schedule is used.")
    parser.add_argument("name", nargs="+", help="player name, e.g. \"Ja'Marr Chase\"")
    parser.add_argument("-w", "--week", type=int, metavar="N",
                        help=f"{TARGET_SEASON} regular-season week to project (default: next unplayed)")
    parser.add_argument("--no-refresh", action="store_true",
                        help=f"use the saved depth chart even if it is over {DEPTH_REFRESH_HOURS}h old")
    parser.add_argument("--model-dir", type=Path, metavar="DIR",
                        help="load models from DIR instead of model/, e.g. a candidate from train.py --model-dir")
    args = parser.parse_args()

    global MODEL_DIR, METADATA_PATH, REFRESH_DEPTH_CHART
    if args.no_refresh:
        REFRESH_DEPTH_CHART = False
    if args.model_dir is not None:
        MODEL_DIR = args.model_dir.resolve()
        METADATA_PATH = MODEL_DIR / "model_metadata.json"

    try:
        predict(" ".join(args.name).strip(), args.week)
    except PredictionError as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
