from pathlib import Path

import numpy as np
import pandas as pd

from layer1_base_join import BOX_STAT_COLS, NGS_START_YEAR

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "base_processed.parquet"
EXPECTED_ROWS = 400_000
NON_NULL_THRESHOLD = 90.0

JOIN_GROUPS = {
    "schedule join": ["game_id", "team", "opponent"],
    "weather join": ["weather_temp", "weather_wind"],
    "vegas join": ["vegas_team_line"],
    "box scores join": ["targets", "receiving_yards", "rushing_yards", "passing_yards"],
    "injury join": ["injury_status"],
    "bio data join": ["bio_age", "bio_position"],
    "offensive coordinators join": ["oc_name"],
    "oline join": ["oline_rank"],
    "sos join": ["preseason_sos", "adjusted_sos"],
}

NGS_PREFIXES = ["ngs_passing_", "ngs_rushing_", "ngs_receiving_"]

STAT_SANITY_COLS = ["targets", "receptions", "receiving_yards", "carries", "rushing_yards", "passing_yards"]
STAT_SANITY_LIMITS = {
    "targets": 30,
    "receptions": 25,
    "receiving_yards": 500,
    "carries": 45,
    "rushing_yards": 500,
    "passing_yards": 550,
}


def check_shape(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 80)
    print("1. SHAPE CHECK")
    print("=" * 80)
    rows, cols = df.shape
    print(f"Total row count: {rows}")
    print(f"Total column count: {cols}")

    ratio = rows / EXPECTED_ROWS
    passed = 0.5 <= ratio <= 2.0
    if not passed:
        print(f"[FLAG] row count {rows} is drastically different from the ~{EXPECTED_ROWS} expected (ratio {ratio:.2f}).")
    else:
        print(f"Row count is within a reasonable range of the ~{EXPECTED_ROWS} expected (ratio {ratio:.2f}).")
    return passed


def check_join_success(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 80)
    print("2. JOIN SUCCESS CHECK")
    print("=" * 80)
    passed = True

    for label, cols in JOIN_GROUPS.items():
        print(f"\n{label}:")
        for col in cols:
            if col not in df.columns:
                print(f"  [MISSING COLUMN] {col}")
                passed = False
                continue
            pct = df[col].notna().mean() * 100
            flag = " [SUSPICIOUS: below 90%]" if pct < NON_NULL_THRESHOLD else ""
            if flag:
                passed = False
            print(f"  {col}: {pct:.1f}% non-null{flag}")

    print("\nngs_* columns (split pre-2016 vs 2016+):")
    pre = df[df["season"] < NGS_START_YEAR]
    post = df[df["season"] >= NGS_START_YEAR]
    for prefix in NGS_PREFIXES:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            print(f"  [MISSING] no columns found with prefix {prefix}")
            passed = False
            continue
        for col in cols:
            pre_pct = pre[col].notna().mean() * 100 if len(pre) else float("nan")
            post_pct = post[col].notna().mean() * 100 if len(post) else float("nan")
            flag = " [SUSPICIOUS: 2016+ below 90%]" if post_pct < NON_NULL_THRESHOLD else ""
            if flag:
                passed = False
            print(f"  {col}: pre-2016 {pre_pct:.1f}% non-null, 2016+ {post_pct:.1f}% non-null{flag}")

    return passed


def check_dnp_flag_logic(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 80)
    print("3. DNP FLAG LOGIC CHECK")
    print("=" * 80)
    passed = True

    out_ir = df[df["injury_status"].isin(["Out", "IR"])]
    print(f"Rows with injury_status Out/IR: {len(out_ir)}")
    if len(out_ir) > 0:
        pct_flagged = (out_ir["dnp_flag"] == 1).mean() * 100
        print(f"  Of those, {pct_flagged:.2f}% have dnp_flag == 1")
        if pct_flagged < 100:
            passed = False
            bad = out_ir[out_ir["dnp_flag"] != 1]
            print(f"  [FAIL] {len(bad)} Out/IR rows do not have dnp_flag == 1")
            print(bad[["player_id", "player_name", "season", "week", "injury_status", "dnp_flag"]].head(10))

    dnp_not_out = df[(df["dnp_flag"] == 1) & (~df["injury_status"].isin(["Out", "IR"]))]
    print(f"\nRows with dnp_flag == 1 but injury_status NOT Out/IR: {len(dnp_not_out)}")
    if len(dnp_not_out) > 0:
        passed = False
        print("[FAIL] expected 0 such rows")
        print(dnp_not_out[["player_id", "player_name", "season", "week", "injury_status", "dnp_flag"]].head(10))

    dnp_rows = df[df["dnp_flag"] == 1]
    present_box_cols = [c for c in BOX_STAT_COLS if c in df.columns]
    bad_mask = (dnp_rows[present_box_cols].isna()) | (dnp_rows[present_box_cols] != 0)
    bad_any = bad_mask.any(axis=1)
    print(f"\nRows with dnp_flag == 1: {len(dnp_rows)}")
    print(f"Of those, rows where some BOX_STAT_COLS are not exactly 0: {bad_any.sum()}")
    if bad_any.sum() > 0:
        passed = False
        offending_cols = bad_mask.loc[bad_any].sum().sort_values(ascending=False)
        offending_cols = offending_cols[offending_cols > 0]
        print("[FAIL] columns with non-zero/NaN values on dnp_flag==1 rows:")
        for col, n in offending_cols.items():
            print(f"  {col}: {n} offending rows")

    return passed


def check_stat_sanity(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 80)
    print("4. STAT SANITY CHECK")
    print("=" * 80)
    passed = True

    present_cols = [c for c in STAT_SANITY_COLS if c in df.columns]
    print(df[present_cols].describe())

    for col in present_cols:
        series = df[col]
        n_negative = (series < 0).sum()
        if n_negative > 0:
            passed = False
            print(f"[FAIL] {col}: {n_negative} rows with negative values")

        limit = STAT_SANITY_LIMITS.get(col)
        if limit is not None:
            n_over = (series > limit).sum()
            if n_over > 0:
                passed = False
                print(f"[FLAG] {col}: {n_over} rows exceed sanity limit of {limit} (max value seen: {series.max()})")

    return passed


def check_spot_coherence(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("5. SPOT-CHECK COHERENCE")
    print("=" * 80)

    counts = df.groupby("player_id").size()
    eligible = counts[counts >= 10].index
    if len(eligible) == 0:
        print("[WARN] no players found with at least 10 weeks of data; skipping spot-check.")
        return

    sample_ids = pd.Series(eligible).sample(min(3, len(eligible)), random_state=None).tolist()
    display_cols = ["season", "week", "team", "opponent", "weather_temp", "targets", "receiving_yards", "dnp_flag", "injury_status"]
    display_cols = [c for c in display_cols if c in df.columns]

    for player_id in sample_ids:
        player_rows = df[df["player_id"] == player_id].sort_values(["season", "week"])
        name = player_rows["player_name"].iloc[0]
        print(f"\nPlayer: {name} ({player_id}), {len(player_rows)} rows")
        print(player_rows[display_cols].to_string(index=False))


def check_duplicates(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 80)
    print("6. DUPLICATE CHECK")
    print("=" * 80)

    counts = df.groupby(["player_id", "season", "week"]).size()
    dupes = counts[counts > 1]
    if len(dupes) == 0:
        print("No duplicate (player_id, season, week) rows found.")
        return True

    print(f"[FAIL] {len(dupes)} duplicate (player_id, season, week) combinations found, {(dupes - 1).sum()} extra rows.")
    print(dupes.sort_values(ascending=False).head(10))
    return False


def check_midseason_trades(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("7. MID-SEASON TRADE CHECK")
    print("=" * 80)

    team_counts = df.groupby(["player_id", "season"])["team"].nunique()
    traded = team_counts[team_counts > 1]
    print(f"Player-seasons with more than one team: {len(traded)}")

    if len(traded) == 0:
        return

    sample_keys = traded.index[: min(10, len(traded))]
    for player_id, season in sample_keys:
        rows = df[(df["player_id"] == player_id) & (df["season"] == season)].sort_values("week")
        name = rows["player_name"].iloc[0]
        print(f"\nPlayer: {name} ({player_id}), season {season}")
        print(rows[["week", "team", "opponent"]].to_string(index=False))

        weeks = rows["week"].tolist()
        gaps = [b - a for a, b in zip(weeks, weeks[1:]) if (b - a) > 1]
        if gaps:
            print(f"  [FLAG] non-consecutive week gaps detected: {gaps}")


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"[fatal] {INPUT_PATH} does not exist. Run layer1_base_join.py first.")
        return

    df = pd.read_parquet(INPUT_PATH)

    results = {}
    results["1. SHAPE CHECK"] = check_shape(df)
    results["2. JOIN SUCCESS CHECK"] = check_join_success(df)
    results["3. DNP FLAG LOGIC CHECK"] = check_dnp_flag_logic(df)
    results["4. STAT SANITY CHECK"] = check_stat_sanity(df)
    check_spot_coherence(df)
    results["6. DUPLICATE CHECK"] = check_duplicates(df)
    check_midseason_trades(df)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
    print("  [INFO] 5. SPOT-CHECK COHERENCE — manual review required, see output above")
    print("  [INFO] 7. MID-SEASON TRADE CHECK — manual review required, see output above")


if __name__ == "__main__":
    main()
