from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
BASE_PATH = PROCESSED_DIR / "base_processed.parquet"
TEAM_AGGS_PATH = PROCESSED_DIR / "team_aggs.parquet"
OUTPUT_PATH = PROCESSED_DIR / "final_training_data.parquet"

DEF_STRENGTH_POSITION_MAP = {
    "def_strength_vs_rb": "RB",
    "def_strength_vs_wr": "WR",
    "def_strength_vs_te": "TE",
    "def_strength_vs_qb": "QB",
}


def safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / denom.replace(0, np.nan)


def build_matchup_features(result: pd.DataFrame, team_aggs: pd.DataFrame) -> pd.DataFrame:
    long_def = team_aggs.melt(
        id_vars=["season", "team"],
        value_vars=list(DEF_STRENGTH_POSITION_MAP),
        var_name="def_col",
        value_name="def_strength",
    )
    long_def["lookup_position"] = long_def["def_col"].map(DEF_STRENGTH_POSITION_MAP)
    long_def = long_def.drop(columns="def_col")

    league_avg = long_def.groupby(["season", "lookup_position"])["def_strength"].mean().reset_index()
    league_avg = league_avg.rename(columns={"def_strength": "league_avg_def_strength", "lookup_position": "lookup_position2"})

    result["position_for_matchup"] = result["bio_position"].fillna(result["position"])

    opp_lookup = long_def.rename(columns={"team": "opponent", "def_strength": "opp_def_strength_for_position"})
    result = result.merge(
        opp_lookup,
        left_on=["season", "opponent", "position_for_matchup"],
        right_on=["season", "opponent", "lookup_position"],
        how="left",
    )
    result = result.drop(columns="lookup_position")

    result = result.merge(
        league_avg,
        left_on=["season", "position_for_matchup"],
        right_on=["season", "lookup_position2"],
        how="left",
    )
    result = result.drop(columns="lookup_position2")

    result["matchup_advantage_score"] = result["opp_def_strength_for_position"] - result["league_avg_def_strength"]
    result = result.drop(columns=["position_for_matchup", "league_avg_def_strength"])
    return result


def build_usage_trend_features(result: pd.DataFrame) -> pd.DataFrame:
    result = result.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    result["usage"] = result["targets"] + result["carries"]
    result["cumulative_usage"] = result.groupby(["player_id", "season"])["usage"].cumsum()

    cum_target_share_sum = result.groupby(["player_id", "season"])["target_share"].cumsum()
    cum_count = result.groupby(["player_id", "season"]).cumcount() + 1
    result["avg_target_share"] = cum_target_share_sum / cum_count

    prior = result[["player_id", "season", "week", "cumulative_usage", "avg_target_share"]].copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(columns={"cumulative_usage": "prior_usage", "avg_target_share": "prior_avg_target_share"})

    result = result.merge(prior, on=["player_id", "season", "week"], how="left")

    result["usage_trend"] = safe_ratio(result["cumulative_usage"], result["prior_usage"])
    result["target_share_trend"] = safe_ratio(result["avg_target_share"], result["prior_avg_target_share"])

    result = result.drop(columns=["usage", "cumulative_usage", "avg_target_share", "prior_usage", "prior_avg_target_share"])
    return result


def build_seasonal_context_features(result: pd.DataFrame) -> pd.DataFrame:
    result["weeks_into_season"] = result["week"]

    result["played"] = (result["dnp_flag"] == 0).astype(int)
    result["games_played_this_szn"] = result.groupby(["player_id", "season"])["played"].cumsum()
    result = result.drop(columns="played")

    team_weeks = result[["season", "team", "week"]].drop_duplicates()

    def find_bye(group: pd.DataFrame):
        weeks = sorted(group["week"].dropna().unique())
        if len(weeks) < 2:
            return np.nan
        full_range = set(range(int(weeks[0]), int(weeks[-1]) + 1))
        missing = full_range - set(int(w) for w in weeks)
        if len(missing) == 1:
            return float(next(iter(missing)))
        return np.nan

    bye_weeks = team_weeks.groupby(["season", "team"]).apply(find_bye).reset_index(name="bye_week")

    result = result.merge(bye_weeks, on=["season", "team"], how="left")
    result["bye_week_passed"] = np.where(
        result["bye_week"].isna(),
        np.nan,
        (result["week"] > result["bye_week"]).astype(float),
    )
    result = result.drop(columns="bye_week")
    return result


def build_injury_recovery_features(result: pd.DataFrame) -> pd.DataFrame:
    result = result.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    result["injury_week_num"] = np.where(result["dnp_flag"] == 1, result["week"], np.nan)
    result["injury_week_num"] = result.groupby(["player_id", "season"])["injury_week_num"].ffill()

    result["weeks_since_injury"] = np.where(
        result["injury_week_num"].notna(),
        result["week"] - result["injury_week_num"],
        99,
    )
    result.loc[result["dnp_flag"] == 1, "weeks_since_injury"] = 0
    result = result.drop(columns="injury_week_num")

    result["prev_injuries_this_szn"] = result.groupby(["player_id", "season"])["dnp_flag"].cumsum()
    return result


def build_rest_days(result: pd.DataFrame) -> pd.DataFrame:
    result["rest_days"] = np.select(
        [result["is_home"] == True, result["is_home"] == False],
        [result["home_rest"], result["away_rest"]],
        default=np.nan,
    )
    return result


def main() -> None:
    base = pd.read_parquet(BASE_PATH)
    team_aggs = pd.read_parquet(TEAM_AGGS_PATH)

    dup_mask = base.duplicated(subset=["player_id", "season", "week"], keep=False)
    if dup_mask.any():
        print(f"[flag] base_processed.parquet has {dup_mask.sum()} rows sharing a duplicate (player_id, season, week) key - a Layer 1 data issue, not a Layer 3 join bug:")
        print(base.loc[dup_mask, ["player_id", "player_name", "season", "week", "team", "opponent", "fantasy_points_ppr"]].to_string())
        base = base.sort_values("fantasy_points_ppr", ascending=False).drop_duplicates(subset=["player_id", "season", "week"], keep="first")
        base = base.sort_index()
        print(f"[flag] Deduplicated by keeping the higher fantasy_points_ppr row per key. Rows remaining: {len(base)}.")

    input_row_count = len(base)

    result = base.merge(team_aggs, on=["season", "team"], how="left")
    if len(result) != input_row_count:
        print(f"[flag] Row count changed after team_aggs merge: {input_row_count} -> {len(result)}. Duplicate keys in team_aggs.")

    result = build_matchup_features(result, team_aggs)
    result = build_usage_trend_features(result)
    result = build_seasonal_context_features(result)
    result = build_injury_recovery_features(result)
    result = build_rest_days(result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("VERIFICATION REPORT")
    print("=" * 80)

    print(f"\nRow count: {len(result)} (expected {input_row_count})")
    if len(result) != input_row_count:
        print("[flag] Row count mismatch vs Layer 1 input.")

    print(f"Column count: {len(result.columns)}")

    new_cols = [
        "opp_def_strength_for_position", "matchup_advantage_score", "usage_trend",
        "target_share_trend", "weeks_into_season", "games_played_this_szn",
        "bye_week_passed", "weeks_since_injury", "prev_injuries_this_szn", "rest_days",
    ]

    expected_ranges = {
        "opp_def_strength_for_position": (3, 15),
        "matchup_advantage_score": (-5, 5),
        "usage_trend": (0.1, 5),
        "target_share_trend": (0.1, 5),
        "weeks_into_season": (1, 18),
        "games_played_this_szn": (1, 17),
        "weeks_since_injury": (0, 99),
        "prev_injuries_this_szn": (0, 17),
        "rest_days": (3, 20),
    }

    for col in new_cols:
        print(f"\n{col} describe():")
        print(result[col].describe())
        if col in expected_ranges:
            lo, hi = expected_ranges[col]
            outliers = result[(result[col].notna()) & ((result[col] < lo) | (result[col] > hi))]
            if not outliers.empty:
                print(f"[flag] {col} has {len(outliers)} rows outside expected range [{lo}, {hi}].")

    bad_bye = result[~result["bye_week_passed"].isin([0, 1]) & result["bye_week_passed"].notna()]
    if not bad_bye.empty:
        print(f"\n[flag] bye_week_passed has {len(bad_bye)} rows with values other than 0/1/NaN.")

    print("\nSpot-check of 3 random players with >= 15 weeks of data:")
    counts = result.groupby("player_id").size()
    eligible = counts[counts >= 15].index
    if len(eligible) > 0:
        sample_players = pd.Series(eligible).sample(min(3, len(eligible)), random_state=None)
        spot_cols = ["season", "week", "team", "opponent", "targets", "dnp_flag", "usage_trend", "weeks_since_injury", "rest_days"]
        for pid in sample_players:
            print("-" * 60)
            print(f"player_id: {pid}")
            player_rows = result[result["player_id"] == pid].sort_values(["season", "week"])
            print(player_rows[spot_cols].to_string(index=False))

    print("\nNull count summary for new columns:")
    for col in new_cols:
        n_null = result[col].isna().sum()
        print(f"  {col}: {n_null} nulls")
    print("  usage_trend/target_share_trend nulls expected for rookies or players with no prior-season data.")
    print("  opp_def_strength_for_position/matchup_advantage_score nulls expected for non-RB/WR/TE/QB positions.")
    print("  bye_week_passed nulls expected when a team's bye week could not be determined from the data.")

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
