from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "base_processed.parquet"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "team_aggs.parquet"


def safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return numer / denom.replace(0, np.nan)


def defensive_strength(df: pd.DataFrame, position: str, numer_col: str, denom_col: str, out_col: str) -> pd.DataFrame:
    subset = df[df["bio_position"] == position]
    grouped = subset.groupby(["season", "opponent"]).agg(
        numer=(numer_col, "sum"),
        denom=(denom_col, "sum"),
    ).reset_index()
    grouped[out_col] = safe_ratio(grouped["numer"], grouped["denom"])
    return grouped.rename(columns={"opponent": "team"})[["season", "team", out_col]]


def defensive_strength_qb(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["bio_position"] == "QB"]
    grouped = subset.groupby(["season", "opponent"]).agg(
        sacks=("sacks", "sum"),
        interceptions=("interceptions", "sum"),
        games=("game_id", "nunique"),
    ).reset_index()
    grouped["def_strength_vs_qb"] = safe_ratio(grouped["sacks"] + grouped["interceptions"], grouped["games"])
    return grouped.rename(columns={"opponent": "team"})[["season", "team", "def_strength_vs_qb"]]


def offensive_share(df: pd.DataFrame, position: str, stat_col: str, out_col: str) -> pd.DataFrame:
    total = df.groupby(["season", "team"])[stat_col].sum().rename("total").reset_index()
    subset = df[df["bio_position"] == position]
    pos_total = subset.groupby(["season", "team"])[stat_col].sum().rename("pos_total").reset_index()
    merged = total.merge(pos_total, on=["season", "team"], how="left")
    merged["pos_total"] = merged["pos_total"].fillna(0)
    merged[out_col] = safe_ratio(merged["pos_total"], merged["total"])
    return merged[["season", "team", out_col]]


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    unique_teams = df.groupby(["season", "team"]).size().reset_index()[["season", "team"]]

    rb_carries = df[(df["bio_position"] == "RB") & (df["carries"] > 0)]
    def_vs_rb = rb_carries.groupby(["season", "opponent"]).agg(
        numer=("rushing_yards", "sum"),
        denom=("carries", "sum"),
    ).reset_index()
    def_vs_rb["def_strength_vs_rb"] = safe_ratio(def_vs_rb["numer"], def_vs_rb["denom"])
    def_vs_rb = def_vs_rb.rename(columns={"opponent": "team"})[["season", "team", "def_strength_vs_rb"]]

    def_vs_wr = defensive_strength(df, "WR", "receiving_yards", "targets", "def_strength_vs_wr")
    def_vs_te = defensive_strength(df, "TE", "receiving_yards", "targets", "def_strength_vs_te")
    def_vs_qb = defensive_strength_qb(df)

    target_share_wr = offensive_share(df, "WR", "targets", "team_target_share_to_wr")
    target_share_te = offensive_share(df, "TE", "targets", "team_target_share_to_te")
    target_share_rb = offensive_share(df, "RB", "targets", "team_target_share_to_rb")
    rush_share_rb = offensive_share(df, "RB", "carries", "team_rush_share_to_rb")

    sos = df.groupby(["season", "team"]).agg(
        preseason_sos=("preseason_sos", "first"),
        adjusted_sos=("adjusted_sos", "first"),
    ).reset_index()

    result = unique_teams
    for piece in (def_vs_rb, def_vs_wr, def_vs_te, def_vs_qb, target_share_wr, target_share_te, target_share_rb, rush_share_rb, sos):
        result = result.merge(piece, on=["season", "team"], how="left")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("VERIFICATION REPORT")
    print("=" * 80)

    print(f"\nRow count: {len(result)} (expected ~512 for 16 seasons x 32 teams)")
    if len(result) != 512:
        print(f"[flag] Row count is {len(result)}, not 512. Likely due to team relocations and/or an incomplete season.")

    describe_cols = ["def_strength_vs_rb", "def_strength_vs_wr", "def_strength_vs_te", "def_strength_vs_qb"]
    print("\nDefensive strength describe():")
    print(result[describe_cols].describe())

    yard_cols = ["def_strength_vs_rb", "def_strength_vs_wr", "def_strength_vs_te"]
    for col in yard_cols:
        outliers = result[(result[col] < 2) | (result[col] > 15)]
        if not outliers.empty:
            print(f"\n[flag] {col} has {len(outliers)} outlier rows (<2 or >15):")
            print(outliers[["season", "team", col]])

    qb_outliers = result[(result["def_strength_vs_qb"] < 0) | (result["def_strength_vs_qb"] > 5)]
    if not qb_outliers.empty:
        print(f"\n[flag] def_strength_vs_qb has {len(qb_outliers)} outlier rows (<0 or >5):")
        print(qb_outliers[["season", "team", "def_strength_vs_qb"]])

    print("\nTarget/rush share describe():")
    share_cols = ["team_target_share_to_wr", "team_target_share_to_te", "team_target_share_to_rb", "team_rush_share_to_rb"]
    print(result[share_cols].describe())

    result["team_target_pct_sum"] = (
        result["team_target_share_to_wr"] + result["team_target_share_to_te"] + result["team_target_share_to_rb"]
    )
    bad_share = result[(result["team_target_pct_sum"] < 0.90) | (result["team_target_pct_sum"] > 1.10)]
    if not bad_share.empty:
        print(f"\n[flag] {len(bad_share)} rows have team_target_pct_sum outside 0.90-1.10:")
        print(bad_share[["season", "team", "team_target_pct_sum"]])
    else:
        print("\nteam_target_pct_sum within 0.90-1.10 for all rows.")

    print("\nSpot-check of 5 random teams:")
    spot_cols = ["season", "team"] + describe_cols + share_cols
    sample = result.sample(min(5, len(result)), random_state=None)
    for _, row in sample.iterrows():
        print("-" * 60)
        for col in spot_cols:
            print(f"  {col}: {row[col]}")

    print("\nSOS describe():")
    print(result[["preseason_sos", "adjusted_sos"]].describe())

    sos_gap = result[(result["adjusted_sos"].notna()) & (result["preseason_sos"].notna())]
    sos_flag = sos_gap[(sos_gap["preseason_sos"] - sos_gap["adjusted_sos"]) > 5]
    if not sos_flag.empty:
        print(f"\n[flag] {len(sos_flag)} rows where adjusted_sos is more than 5 below preseason_sos:")
        print(sos_flag[["season", "team", "preseason_sos", "adjusted_sos"]])
    else:
        print("\nNo rows where adjusted_sos falls more than 5 below preseason_sos.")

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
