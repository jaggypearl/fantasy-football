import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

METADATA_PATH = MODEL_DIR / "model_metadata.json"
SCHEDULE_PATH = RAW_DIR / "schedules" / "schedules_2026.csv"
BIO_PATH = RAW_DIR / "bio_data" / "bio_data_2026.csv"
OC_PATH = RAW_DIR / "offensive_coordinators" / "offensive_coordinators_2026.csv"
SOS_PATH = RAW_DIR / "strength_of_schedule" / "strength_of_schedule_2026.csv"
TEAM_AGGS_PATH = PROCESSED_DIR / "team_aggs.parquet"
TRAINING_PATH = PROCESSED_DIR / "final_training_data.parquet"

TARGET_SEASON = 2026
CONTEXT_SEASON = 2025
TARGET_WEEK = 1

DEFAULT_WEATHER_TEMP = 70.0
DEFAULT_WEATHER_WIND = 5.0
DEFAULT_SPREAD_LINE = 0.0
DEFAULT_TOTAL_LINE = 45.0

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
    ranked = ranked.sort_values("_rank", ascending=False)
    return ranked.iloc[0]


def find_game(schedule: pd.DataFrame, team: str) -> tuple[pd.Series, bool]:
    week = schedule[schedule["week"] == TARGET_WEEK].copy()
    week["home_team"] = week["home_team"].map(normalize_team)
    week["away_team"] = week["away_team"].map(normalize_team)

    home = week[week["home_team"] == team]
    if not home.empty:
        return home.iloc[0], True

    away = week[week["away_team"] == team]
    if not away.empty:
        return away.iloc[0], False

    fail(f"no week {TARGET_WEEK} game found for {team} in {SCHEDULE_PATH.name} (bye or missing schedule row)")


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


def build_overrides(game: pd.Series, is_home: bool, bio: pd.Series, team: str, opponent: str,
                    position: str, oc: pd.Series | None, sos: pd.Series | None,
                    team_def: pd.Series | None, opp_def: pd.Series | None,
                    league_avg: float | None) -> dict[str, float]:
    spread = game.get("spread_line")
    total = game.get("total_line")
    spread = DEFAULT_SPREAD_LINE if pd.isna(spread) else float(spread)
    total = DEFAULT_TOTAL_LINE if pd.isna(total) else float(total)

    home_rest = game.get("home_rest")
    away_rest = game.get("away_rest")
    rest = home_rest if is_home else away_rest

    values: dict[str, float] = {
        "week": float(TARGET_WEEK),
        "weeks_into_season": 1.0,
        "games_played_this_szn": 0.0,
        "bye_week_passed": 0.0,
        "prev_injuries_this_szn": 0.0,
        "is_home": 1.0 if is_home else 0.0,
        "spread_line": spread,
        "total_line": total,
        "implied_home_total": (total - spread) / 2,
        "implied_away_total": (total + spread) / 2,
        "weather_temp": DEFAULT_WEATHER_TEMP,
        "weather_wind": DEFAULT_WEATHER_WIND,
        "sched_temp": DEFAULT_WEATHER_TEMP,
        "sched_wind": DEFAULT_WEATHER_WIND,
        "rest_days": float(rest) if pd.notna(rest) else 7.0,
    }

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
        for column in ("team_target_share_to_wr", "team_target_share_to_te",
                       "team_target_share_to_rb", "team_rush_share_to_rb"):
            value = oc.get(column)
            if pd.notna(value):
                values[column] = float(value)

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


def predict(name: str) -> None:
    metadata = load_json(METADATA_PATH)
    bio = load_csv(BIO_PATH, "2026 bio data")
    schedule = load_csv(SCHEDULE_PATH, "2026 schedule")

    player = find_player(bio, name)
    position = str(player.get("position", "")).upper()
    player_name = player["player_name"]

    if position not in metadata["positions"]:
        fail(f"{player_name} plays {position or 'an unknown position'}; "
             f"models exist only for {', '.join(metadata['positions'])}")

    team = normalize_team(player.get("team"))
    if team is None:
        fail(f"{player_name} has no {TARGET_SEASON} team assigned")

    game, is_home = find_game(schedule, team)
    opponent = normalize_team(game["away_team"] if is_home else game["home_team"])

    position_meta = metadata["positions"][position]
    reference = next(iter(position_meta.values()))
    feature_cols = reference["features"]
    scaler_mean = reference["scaler_mean"]
    scaler_scale = reference["scaler_scale"]

    training = pd.read_parquet(
        TRAINING_PATH,
        columns=sorted(set(feature_cols) | {"season", "week", "player_id", "player_name",
                                            "bio_position", "dnp_flag"}),
        filters=[("season", "==", float(CONTEXT_SEASON))],
    )
    context, context_source = player_context_row(training, player.get("player_id"), player_name,
                                                 position, feature_cols)

    oc = lookup_row(load_csv(OC_PATH, "2026 coordinator data"), "team", team)
    sos = lookup_row(load_csv(SOS_PATH, "2026 strength of schedule"), "team", team)

    team_def = opp_def = None
    league_avg = None
    if TEAM_AGGS_PATH.exists():
        aggs = pd.read_parquet(TEAM_AGGS_PATH)
        aggs = aggs[aggs["season"] == float(CONTEXT_SEASON)]
        team_def = lookup_row(aggs, "team", team)
        opp_def = lookup_row(aggs, "team", opponent)
        if position in DEF_STRENGTH_COLS and not aggs.empty:
            league_avg = aggs[DEF_STRENGTH_COLS[position]].mean()

    overrides = build_overrides(game, is_home, player, team, opponent, position,
                                oc, sos, team_def, opp_def, league_avg)
    raw_vector, defaulted = assemble_vector(feature_cols, context, overrides, scaler_mean)
    normalized = (raw_vector - np.asarray(scaler_mean)) / np.asarray(scaler_scale)
    normalized = normalized.reshape(1, -1)

    models = load_models(position_meta)
    predictions = {stat: max(0.0, float(model.predict(normalized)[0])) for stat, model in models.items()}

    weights = metadata["ppr_weights"]
    points = sum(predictions.get(stat, 0.0) * weight for stat, weight in weights.items())
    points = max(0.0, points)

    location = "vs" if is_home else "at"
    print(f"{player_name} ({position}, {team} {location} {opponent}, Week {TARGET_WEEK})")
    print(f"Projected: {points:.1f} PPR")
    detail = format_detail(position, predictions)
    if detail:
        print(f"({detail})")
    print(f"[context: {context_source}; {len(defaulted)} of {len(feature_cols)} features unavailable]")


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: python prediction.py 'Player Name'")
        sys.exit(1)

    try:
        predict(" ".join(sys.argv[1:]).strip())
    except PredictionError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
