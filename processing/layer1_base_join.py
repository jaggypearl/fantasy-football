from pathlib import Path

import numpy as np
import pandas as pd

from pbp_box_scores import build_pbp_box_scores

START_YEAR = 2010
END_YEAR = 2026
NGS_START_YEAR = 2016

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "base_processed.parquet"

BOX_STAT_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost", "passing_air_yards",
    "passing_yards_after_catch", "passing_first_downs", "passing_epa",
    "passing_2pt_conversions", "pacr", "dakota", "carries", "rushing_yards",
    "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost", "rushing_first_downs",
    "rushing_epa", "rushing_2pt_conversions", "receptions", "targets",
    "receiving_yards", "receiving_tds", "receiving_fumbles", "receiving_fumbles_lost",
    "receiving_air_yards", "receiving_yards_after_catch", "receiving_first_downs",
    "receiving_epa", "receiving_2pt_conversions", "racr", "target_share",
    "air_yards_share", "wopr", "special_teams_tds", "fantasy_points",
    "fantasy_points_ppr",
]
PBP_STAT_COLS = ["pbp_target_share", "pbp_air_yards", "snap_count", "snap_pct"]

TEAM_RELOCATIONS = {
    "OAK": ("OAK", "LV", 2020),
    "LV": ("OAK", "LV", 2020),
    "SD": ("SD", "LAC", 2017),
    "LAC": ("SD", "LAC", 2017),
    "STL": ("STL", "LA", 2016),
    "LA": ("STL", "LA", 2016),
}


def normalize_team_abbr(series: pd.Series, season: pd.Series) -> pd.Series:
    old = series.map(lambda t: TEAM_RELOCATIONS[t][0] if t in TEAM_RELOCATIONS else t)
    new = series.map(lambda t: TEAM_RELOCATIONS[t][1] if t in TEAM_RELOCATIONS else t)
    cutover = series.map(lambda t: TEAM_RELOCATIONS[t][2] if t in TEAM_RELOCATIONS else None)
    return np.where(cutover.notna() & (season >= cutover), new, old)


def load_year_csvs(dir_path: Path, prefix: str, years=range(START_YEAR, END_YEAR + 1), **read_kwargs) -> pd.DataFrame:
    frames = []
    for year in years:
        path = dir_path / f"{prefix}_{year}.csv"
        if not path.exists():
            print(f"[missing] {path} does not exist; skipping {year}.")
            continue
        df = pd.read_csv(path, **read_kwargs)
        if df.empty:
            print(f"[empty] {path} has no rows; skipping {year}.")
            continue
        if "season" not in df.columns:
            df["season"] = year
        frames.append(df)
    if not frames:
        print(f"[fatal] No data loaded for prefix '{prefix}' in {dir_path}.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_schedules() -> pd.DataFrame:
    schedules = load_year_csvs(RAW_DIR / "schedules", "schedules")
    if schedules.empty:
        return schedules
    schedules = schedules[schedules["game_type"] == "REG"].copy()
    schedules["implied_home_total"] = (schedules["total_line"] - schedules["spread_line"]) / 2
    schedules["implied_away_total"] = (schedules["total_line"] + schedules["spread_line"]) / 2
    schedules = schedules.rename(columns={"roof": "sched_roof", "surface": "sched_surface", "temp": "sched_temp", "wind": "sched_wind"})
    return schedules


def build_team_games(schedules: pd.DataFrame) -> pd.DataFrame:
    home = schedules.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = True

    away = schedules.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["is_home"] = False

    team_games = pd.concat([home, away], ignore_index=True)
    return team_games


def load_weather() -> pd.DataFrame:
    weather = load_year_csvs(RAW_DIR / "weather", "weather")
    if weather.empty:
        return weather
    weather = weather.drop(columns=["gameday", "away_team", "home_team", "season"], errors="ignore")
    weather = weather.rename(columns={"roof": "weather_roof", "surface": "weather_surface", "temp": "weather_temp", "wind": "weather_wind"})
    return weather.drop_duplicates(subset=["game_id"])


def load_vegas_lines() -> pd.DataFrame:
    vegas = load_year_csvs(RAW_DIR / "vegas_lines", "vegas_lines")
    if vegas.empty:
        print("[warn] vegas_lines raw data is empty for every season attempted; vegas_team_line will be entirely NaN.")
        return vegas
    vegas = vegas.rename(columns={"side": "team", "line": "vegas_team_line"})
    return vegas[["season", "week", "team", "vegas_team_line"]].drop_duplicates(subset=["season", "week", "team"])


def load_box_scores() -> pd.DataFrame:
    box = load_year_csvs(RAW_DIR / "box_scores", "box_scores")
    if box.empty:
        return box
    box = box[box["season_type"] == "REG"].copy()
    box = box.rename(columns={
        "recent_team": "box_team",
        "opponent_team": "box_opponent_team",
        "player_name": "player_name_short",
        "player_display_name": "player_name",
    })
    box = box.drop(columns=["season_type"], errors="ignore")
    box["box_team"] = normalize_team_abbr(box["box_team"], box["season"])
    box["box_opponent_team"] = normalize_team_abbr(box["box_opponent_team"], box["season"])
    return fill_missing_seasons_from_pbp(box)


def fill_missing_seasons_from_pbp(box: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the stat line from play-by-play for seasons with no box score file.

    nflverse publishes weekly box scores a season behind play-by-play, so the most
    recent season would otherwise join as all-NaN and then be zero-filled into a
    season that looks complete but scores nothing. Validated against 2024, where
    both sources exist: fantasy_points_ppr matches exactly on 99.4% of player-weeks.
    """
    have = set(box["season"].dropna().astype(int).unique())
    missing = [y for y in range(START_YEAR, END_YEAR + 1) if y not in have]
    if not missing:
        return box

    frames = [box]
    for year in missing:
        derived = build_pbp_box_scores(year)
        if derived.empty:
            print(f"[missing] no play-by-play to rebuild box scores for {year}.")
            continue
        derived["box_team"] = normalize_team_abbr(derived["box_team"], derived["season"])
        print(f"[info] {year} has no box_scores CSV; rebuilt {len(derived)} player-weeks from play-by-play.")
        frames.append(derived)

    return pd.concat(frames, ignore_index=True)


def load_injury_reports() -> pd.DataFrame:
    injuries = load_year_csvs(RAW_DIR / "injury_reports", "injury_reports")
    if injuries.empty:
        return injuries
    injuries = injuries[injuries["game_type"] == "REG"].copy()
    injuries = injuries.rename(columns={"gsis_id": "player_id", "report_status": "injury_status", "full_name": "injury_full_name"})
    injuries = injuries.drop(columns=["game_type", "season_type", "position", "first_name", "last_name"], errors="ignore")
    injuries = injuries.drop_duplicates(subset=["player_id", "season", "week"])
    return injuries


def load_bio_data() -> pd.DataFrame:
    bio = load_year_csvs(RAW_DIR / "bio_data", "bio_data")
    if bio.empty:
        return bio
    bio = bio.sort_values("week").drop_duplicates(subset=["player_id", "season"], keep="last")
    bio = bio.drop(columns=["week", "game_type"], errors="ignore")
    rename_map = {c: f"bio_{c}" for c in bio.columns if c not in ("player_id", "season")}
    bio = bio.rename(columns=rename_map)
    return bio


def load_offensive_coordinators() -> pd.DataFrame:
    oc = load_year_csvs(RAW_DIR / "offensive_coordinators", "offensive_coordinators")
    if oc.empty:
        return oc
    return oc.rename(columns={"offensive_coordinator": "oc_name"}).drop_duplicates(subset=["season", "team"])


def load_oline_rankings() -> pd.DataFrame:
    oline = load_year_csvs(RAW_DIR / "oline_rankings_weekly", "oline_rankings_weekly")
    if oline.empty:
        return oline
    oline["team"] = normalize_team_abbr(oline["team"], oline["season"])
    return oline.drop_duplicates(subset=["season", "week", "team"])


def load_strength_of_schedule() -> pd.DataFrame:
    sos = load_year_csvs(RAW_DIR / "strength_of_schedule", "strength_of_schedule")
    if sos.empty:
        return sos
    return sos.drop_duplicates(subset=["season", "team"])


def load_ngs(stat_type: str) -> pd.DataFrame:
    ngs = load_year_csvs(RAW_DIR / "ngs" / stat_type, f"ngs_{stat_type}", years=range(NGS_START_YEAR, END_YEAR + 1))
    if ngs.empty:
        return ngs
    ngs = ngs[(ngs["season_type"] == "REG") & (ngs["week"] > 0)].copy()
    ngs = ngs.rename(columns={"player_gsis_id": "player_id"})
    drop_cols = [
        "season_type", "player_display_name", "player_position", "team_abbr",
        "player_first_name", "player_last_name", "player_jersey_number", "player_short_name",
    ]
    ngs = ngs.drop(columns=drop_cols, errors="ignore")
    rename_map = {c: f"ngs_{stat_type}_{c}" for c in ngs.columns if c not in ("player_id", "season", "week")}
    ngs = ngs.rename(columns=rename_map)
    return ngs.drop_duplicates(subset=["player_id", "season", "week"])


def aggregate_play_by_play() -> pd.DataFrame:
    base_cols = ["season", "week", "posteam", "play_type", "pass_attempt", "rush_attempt", "receiver_player_id", "air_yards"]
    year_frames = []

    for year in range(START_YEAR, END_YEAR + 1):
        path = RAW_DIR / "play_by_play" / f"play_by_play_{year}.csv"
        if not path.exists():
            print(f"[missing] {path} does not exist; skipping {year} for play-by-play aggregation.")
            continue

        header = pd.read_csv(path, nrows=0).columns.tolist()
        has_offense_players = "offense_players" in header
        usecols = list(base_cols)
        if has_offense_players:
            usecols.append("offense_players")
        else:
            print(f"[info] offense_players column not present in play_by_play_{year}.csv; snap_count/snap_pct will be NaN for {year}.")

        pbp = pd.read_csv(path, usecols=usecols, low_memory=False)
        pbp = pbp.dropna(subset=["posteam"])

        team_plays = pbp.groupby(["season", "week", "posteam"]).agg(
            team_pass_attempts=("pass_attempt", "sum"),
            team_offensive_plays=("play_type", lambda s: s.isin(["pass", "run"]).sum()),
        ).reset_index()

        targets = pbp[pbp["pass_attempt"] == 1].dropna(subset=["receiver_player_id"])
        target_stats = targets.groupby(["season", "week", "receiver_player_id", "posteam"]).agg(
            pbp_targets=("pass_attempt", "sum"),
            pbp_air_yards=("air_yards", "sum"),
        ).reset_index().rename(columns={"receiver_player_id": "player_id"})

        target_stats = target_stats.merge(team_plays[["season", "week", "posteam", "team_pass_attempts"]], on=["season", "week", "posteam"], how="left")
        target_stats["pbp_target_share"] = target_stats["pbp_targets"] / target_stats["team_pass_attempts"]
        target_stats = target_stats.drop(columns=["posteam", "pbp_targets", "team_pass_attempts"])

        if has_offense_players:
            scrimmage = pbp[pbp["play_type"].isin(["pass", "run"])]
            snaps = scrimmage.dropna(subset=["offense_players"])[["season", "week", "posteam", "offense_players"]].copy()
            snaps["offense_players"] = snaps["offense_players"].str.split(";")
            snaps = snaps.explode("offense_players").rename(columns={"offense_players": "player_id"})
            snap_stats = snaps.groupby(["season", "week", "player_id"]).size().reset_index(name="snap_count")
            team_snap_map = snaps.drop_duplicates(subset=["season", "week", "player_id"])[["season", "week", "player_id", "posteam"]]
            snap_stats = snap_stats.merge(team_snap_map, on=["season", "week", "player_id"], how="left")
            snap_stats = snap_stats.merge(team_plays[["season", "week", "posteam", "team_offensive_plays"]], on=["season", "week", "posteam"], how="left")
            snap_stats["snap_pct"] = snap_stats["snap_count"] / snap_stats["team_offensive_plays"]
            snap_stats = snap_stats.drop(columns=["posteam", "team_offensive_plays"])
        else:
            snap_stats = pd.DataFrame(columns=["season", "week", "player_id", "snap_count", "snap_pct"])

        year_agg = target_stats.merge(snap_stats, on=["season", "week", "player_id"], how="outer")
        year_frames.append(year_agg)
        print(f"Aggregated play-by-play for {year}: {len(year_agg)} player-week rows.")

    if not year_frames:
        print("[fatal] No play-by-play data aggregated for any year.")
        return pd.DataFrame(columns=["season", "week", "player_id"] + PBP_STAT_COLS)

    pbp_agg = pd.concat(year_frames, ignore_index=True)
    pbp_agg["routes_run"] = np.nan
    print("[info] play_by_play does not contain route-participation data, so routes_run cannot be computed and is left as NaN for all rows.")
    return pbp_agg


def build_universe(box_scores: pd.DataFrame, injury_reports: pd.DataFrame) -> pd.DataFrame:
    box_universe = box_scores[["player_id", "season", "week", "box_team"]].rename(columns={"box_team": "team"}).drop_duplicates()
    box_universe["_src"] = 0

    injury_universe = injury_reports[["player_id", "season", "week", "team"]].drop_duplicates()
    injury_universe["_src"] = 1

    universe = pd.concat([box_universe, injury_universe], ignore_index=True)
    universe = universe.sort_values("_src").drop_duplicates(subset=["player_id", "season", "week"], keep="first")
    universe = universe.drop(columns="_src").dropna(subset=["team"])

    print(
        f"[info] No dedicated weekly-roster dataset exists in data/raw/. Player-week universe was built from the "
        f"union of box_scores and injury_reports appearances ({len(universe)} rows). Players who neither recorded "
        f"a stat nor appeared on an injury report in a given week (e.g. healthy inactive depth players) are not "
        f"represented, so the true ~1,500 eligible-players-per-week figure will likely be undercounted."
    )
    return universe


def build_name_lookup(box_scores: pd.DataFrame, bio_data: pd.DataFrame, injury_reports: pd.DataFrame) -> pd.DataFrame:
    box_names = box_scores[["player_id", "player_name"]].dropna().drop_duplicates(subset="player_id")
    bio_names = bio_data[["player_id", "bio_player_name"]].rename(columns={"bio_player_name": "player_name"}).dropna().drop_duplicates(subset="player_id") if "bio_player_name" in bio_data.columns else pd.DataFrame(columns=["player_id", "player_name"])
    injury_names = injury_reports[["player_id", "injury_full_name"]].rename(columns={"injury_full_name": "player_name"}).dropna().drop_duplicates(subset="player_id") if "injury_full_name" in injury_reports.columns else pd.DataFrame(columns=["player_id", "player_name"])

    combined = pd.concat([box_names, bio_names, injury_names], ignore_index=True)
    return combined.drop_duplicates(subset="player_id", keep="first")


def track_new_columns(column_origin: dict, source: str, before_cols, after_cols) -> None:
    new_cols = [c for c in after_cols if c not in before_cols]
    for col in new_cols:
        column_origin[col] = source


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    column_origin: dict = {}

    print("Loading schedules...")
    schedules = load_schedules()
    if schedules.empty:
        print("[fatal] Cannot proceed without schedule data.")
        return
    team_games = build_team_games(schedules)

    print("Loading weather...")
    weather = load_weather()
    if not weather.empty:
        team_games = team_games.merge(weather, on="game_id", how="left")

    print("Loading vegas_lines...")
    vegas = load_vegas_lines()
    if not vegas.empty:
        team_games = team_games.merge(vegas, on=["season", "week", "team"], how="left")
    else:
        team_games["vegas_team_line"] = np.nan

    print("Loading box_scores...")
    box_scores = load_box_scores()
    if box_scores.empty:
        print("[fatal] Cannot proceed without box_scores data.")
        return

    print("Loading injury_reports...")
    injury_reports = load_injury_reports()
    if injury_reports.empty:
        print("[warn] injury_reports data is empty; dnp_flag will default to 0 for all rows and injury columns will be NaN.")

    print("Building player-week universe...")
    universe = build_universe(box_scores, injury_reports)

    print("Building player name lookup...")
    bio_data = load_bio_data()
    name_lookup = build_name_lookup(box_scores, bio_data, injury_reports)

    base = universe.merge(name_lookup, on="player_id", how="left")

    print("Joining schedule/team context (game_id, opponent, weather, vegas)...")
    cols_before = list(base.columns)
    base = base.merge(team_games, on=["season", "week", "team"], how="left")
    track_new_columns(column_origin, "schedule/weather/vegas join (game_id unmatched for team-week)", cols_before, base.columns)

    missing_game = base["game_id"].isna().sum()
    if missing_game:
        print(f"[warn] {missing_game} player-week rows could not be matched to a schedule game for their team/week combination.")

    print("Joining box_scores stats...")
    box_join_cols = [c for c in box_scores.columns if c not in ("player_id", "season", "week", "box_team", "player_name")]
    cols_before = list(base.columns)
    base = base.merge(box_scores[["player_id", "season", "week"] + box_join_cols], on=["player_id", "season", "week"], how="left")
    track_new_columns(column_origin, "box_scores join (player had no box_scores row that week)", cols_before, base.columns)

    print("Aggregating and joining play-by-play derived stats...")
    pbp_agg = aggregate_play_by_play()
    cols_before = list(base.columns)
    base = base.merge(pbp_agg, on=["player_id", "season", "week"], how="left")
    track_new_columns(column_origin, "play-by-play aggregation join (no matching plays that week)", cols_before, base.columns)

    print("Joining NGS passing/rushing/receiving...")
    for stat_type in ("passing", "rushing", "receiving"):
        ngs = load_ngs(stat_type)
        if ngs.empty:
            print(f"[warn] No NGS {stat_type} data loaded; ngs_{stat_type}_* columns will be entirely NaN.")
            continue
        cols_before = list(base.columns)
        base = base.merge(ngs, on=["player_id", "season", "week"], how="left")
        track_new_columns(column_origin, f"ngs_{stat_type} join (pre-{NGS_START_YEAR} or no NGS record that week)", cols_before, base.columns)

    print("Joining injury_reports detail columns...")
    if not injury_reports.empty:
        injury_detail = injury_reports.rename(columns={"team": "injury_team"})
        cols_before = list(base.columns)
        base = base.merge(injury_detail, on=["player_id", "season", "week"], how="left")
        track_new_columns(column_origin, "injury_reports join (player not on injury report that week)", cols_before, base.columns)
    else:
        base["injury_status"] = np.nan
        column_origin["injury_status"] = "injury_reports join (dataset empty)"

    print("Joining bio_data...")
    if not bio_data.empty:
        cols_before = list(base.columns)
        base = base.merge(bio_data, on=["player_id", "season"], how="left")
        track_new_columns(column_origin, "bio_data join (player not found in seasonal roster snapshot)", cols_before, base.columns)

    print("Joining offensive_coordinators...")
    oc = load_offensive_coordinators()
    if not oc.empty:
        cols_before = list(base.columns)
        base = base.merge(oc, on=["season", "team"], how="left")
        track_new_columns(column_origin, "offensive_coordinators join (no OC record for team-season)", cols_before, base.columns)

    print("Joining oline_rankings_weekly...")
    oline = load_oline_rankings()
    if not oline.empty:
        cols_before = list(base.columns)
        base = base.merge(oline, on=["season", "week", "team"], how="left")
        track_new_columns(column_origin, "oline_rankings_weekly join (no ranking for team-week)", cols_before, base.columns)

    print("Joining strength_of_schedule...")
    sos = load_strength_of_schedule()
    if not sos.empty:
        cols_before = list(base.columns)
        base = base.merge(sos, on=["season", "team"], how="left")
        track_new_columns(column_origin, "strength_of_schedule join (no SOS record for team-season)", cols_before, base.columns)

    print("Computing dnp_flag and filling stat columns...")
    base["injury_status"] = base["injury_status"].fillna("")
    base["dnp_flag"] = base["injury_status"].isin(["Out", "IR"]).astype(int)

    present_box_stat_cols = [c for c in BOX_STAT_COLS if c in base.columns]
    for col in present_box_stat_cols:
        base[col] = base[col].fillna(0)

    present_pbp_stat_cols = [c for c in PBP_STAT_COLS if c in base.columns]
    for col in present_pbp_stat_cols:
        base[col] = base[col].fillna(0)

    ngs_stat_cols = [c for c in base.columns if c.startswith("ngs_passing_") or c.startswith("ngs_rushing_") or c.startswith("ngs_receiving_")]
    ngs_era_mask = base["season"] >= NGS_START_YEAR
    for col in ngs_stat_cols:
        base.loc[ngs_era_mask, col] = base.loc[ngs_era_mask, col].fillna(0)
    print(
        f"[info] ngs_* columns were only zero-filled for seasons >= {NGS_START_YEAR} (when NGS data exists). "
        f"Rows before {NGS_START_YEAR} keep NaN for ngs_* columns because the source data does not exist for that era, "
        f"not because a specific player-week is missing."
    )

    base["fumbles"] = base.get("sack_fumbles", 0) + base.get("rushing_fumbles", 0) + base.get("receiving_fumbles", 0)

    base.loc[base["injury_status"] == "", "injury_status"] = np.nan

    print(f"Saving to {OUTPUT_PATH}...")
    base.to_parquet(OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("VERIFICATION REPORT")
    print("=" * 80)

    print(f"\nTotal row count: {len(base)}")
    print(f"Total column count: {len(base.columns)}")
    print("\nColumn names:")
    for col in base.columns:
        print(f"  {col}")

    print("\nSpot-check of 5 random rows:")
    sample_cols = [
        "player_id", "player_name", "season", "week", "team", "opponent", "game_id",
        "weather_temp", "weather_wind", "spread_line", "total_line",
        "targets", "receptions", "rushing_yards", "passing_yards", "dnp_flag", "injury_status",
    ]
    sample_cols = [c for c in sample_cols if c in base.columns]
    sample = base.sample(min(5, len(base)), random_state=None)
    for _, row in sample.iterrows():
        print("-" * 60)
        for col in sample_cols:
            print(f"  {col}: {row[col]}")

    dnp_counts = base["dnp_flag"].value_counts()
    print(f"\ndnp_flag counts:\n{dnp_counts}")

    print("\nColumns with NaN values remaining after fill logic, grouped by cause:")
    always_expected_nan = {"routes_run": "play_by_play has no route-participation data; left NaN for all rows"}
    nan_by_origin: dict = {}
    for col in base.columns:
        n_missing = base[col].isna().sum()
        if n_missing == 0:
            continue
        if col in always_expected_nan:
            reason = always_expected_nan[col]
        else:
            reason = column_origin.get(col, "unexplained by any tracked join — investigate")
        nan_by_origin.setdefault(reason, []).append((col, n_missing))

    for reason, cols in nan_by_origin.items():
        flagged = "UNEXPECTED" if reason.startswith("unexplained") else "known-gap"
        print(f"\n  [{flagged}] cause: {reason}")
        for col, n_missing in sorted(cols, key=lambda x: -x[1]):
            print(f"    {col}: {n_missing} rows missing")


if __name__ == "__main__":
    main()
