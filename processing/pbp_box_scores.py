"""Derive player-week box score stats from play-by-play.

nflverse publishes weekly box scores a season behind the play-by-play feed, so the
most recent season arrives with play-by-play only. This module rebuilds the same
player-week stat line from the raw plays, letting Layer 1 fill seasons that have no
box_scores_<year>.csv.

Run directly to validate against a season that has both sources:
    python processing/pbp_box_scores.py --validate 2024
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Columns read from the play-by-play CSV. Kept explicit because the full file is
# 397 columns and ~140MB per season.
PBP_USECOLS = [
    "season", "week", "season_type", "posteam", "play_type",
    "passer_player_id", "receiver_player_id", "rusher_player_id",
    "pass_attempt", "rush_attempt", "complete_pass", "interception", "sack",
    "passing_yards", "receiving_yards", "rushing_yards", "air_yards",
    "yards_after_catch", "yards_gained",
    "pass_touchdown", "rush_touchdown", "return_touchdown",
    "first_down_pass", "first_down_rush",
    "fumble", "fumble_lost", "fumbled_1_player_id",
    "two_point_attempt", "two_point_conv_result",
    "punt_returner_player_id", "kickoff_returner_player_id",
    "epa",
]

# Standard PPR scoring.
PPR_WEIGHTS = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "interceptions": -2.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "receptions": 1.0,
    "special_teams_tds": 6.0,
}
FUMBLE_LOST_POINTS = -2.0
TWO_POINT_POINTS = 2.0

KEY = ["player_id", "season", "week"]


def _sum_by(frame: pd.DataFrame, player_col: str, aggs: dict) -> pd.DataFrame:
    """Group plays by (player, season, week) and sum the requested columns."""
    sub = frame.dropna(subset=[player_col])
    if sub.empty:
        return pd.DataFrame(columns=KEY + list(aggs))
    out = sub.groupby([player_col, "season", "week"], dropna=True).agg(**aggs).reset_index()
    return out.rename(columns={player_col: "player_id"})


def pass_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Passes actually thrown - sacks carry pass_attempt = 1 but are not attempts."""
    return pbp[(pbp["pass_attempt"] == 1) & (pbp["sack"] != 1)]


def _passing_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    # nflverse sets pass_attempt = 1 on sacks, but box scores count a sack as a
    # dropback rather than an attempt, so attempts are summed over non-sack plays.
    dropbacks = pbp[(pbp["pass_attempt"] == 1) | (pbp["sack"] == 1)]
    stats = _sum_by(dropbacks, "passer_player_id", {
        "completions": ("complete_pass", "sum"),
        "passing_yards": ("passing_yards", "sum"),
        "passing_tds": ("pass_touchdown", "sum"),
        "interceptions": ("interception", "sum"),
        "sacks": ("sack", "sum"),
        "passing_air_yards": ("air_yards", "sum"),
        "passing_yards_after_catch": ("yards_after_catch", "sum"),
        "passing_first_downs": ("first_down_pass", "sum"),
        "passing_epa": ("epa", "sum"),
    })

    attempts = _sum_by(pass_plays(pbp), "passer_player_id", {"attempts": ("pass_attempt", "sum")})
    stats = stats.merge(attempts, on=KEY, how="left") if not attempts.empty else stats.assign(attempts=0.0)

    # Sack yardage is negative yards_gained on sack plays; box scores report it positive.
    sacks = pbp[pbp["sack"] == 1]
    sack_yards = _sum_by(sacks, "passer_player_id", {"sack_yards": ("yards_gained", "sum")})
    if not sack_yards.empty:
        sack_yards["sack_yards"] = -sack_yards["sack_yards"]
        stats = stats.merge(sack_yards, on=KEY, how="left")
    else:
        stats["sack_yards"] = 0.0
    return stats


def _receiving_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    return _sum_by(pass_plays(pbp), "receiver_player_id", {
        "targets": ("pass_attempt", "sum"),
        "receptions": ("complete_pass", "sum"),
        "receiving_yards": ("receiving_yards", "sum"),
        "receiving_tds": ("pass_touchdown", "sum"),
        "receiving_air_yards": ("air_yards", "sum"),
        "receiving_yards_after_catch": ("yards_after_catch", "sum"),
        "receiving_first_downs": ("first_down_pass", "sum"),
        "receiving_epa": ("epa", "sum"),
    })


def _rushing_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    rushes = pbp[pbp["rush_attempt"] == 1]
    return _sum_by(rushes, "rusher_player_id", {
        "carries": ("rush_attempt", "sum"),
        "rushing_yards": ("rushing_yards", "sum"),
        "rushing_tds": ("rush_touchdown", "sum"),
        "rushing_first_downs": ("first_down_rush", "sum"),
        "rushing_epa": ("epa", "sum"),
    })


def _fumble_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Split fumbles into sack/rushing/receiving buckets the way box scores do."""
    fumbles = pbp[(pbp["fumble"] == 1) & pbp["fumbled_1_player_id"].notna()].copy()
    if fumbles.empty:
        return pd.DataFrame(columns=KEY)

    fumbles["bucket"] = np.select(
        [fumbles["sack"] == 1, fumbles["rush_attempt"] == 1, fumbles["complete_pass"] == 1],
        ["sack", "rushing", "receiving"],
        default="other",
    )
    fumbles = fumbles[fumbles["bucket"] != "other"]
    if fumbles.empty:
        return pd.DataFrame(columns=KEY)

    fumbles = fumbles.rename(columns={"fumbled_1_player_id": "player_id"})
    counts = fumbles.groupby(["player_id", "season", "week", "bucket"]).agg(
        n=("fumble", "sum"), n_lost=("fumble_lost", "sum")
    ).reset_index()

    wide = counts.pivot_table(
        index=KEY, columns="bucket", values=["n", "n_lost"], fill_value=0
    )
    wide.columns = [
        f"{bucket}_fumbles" if metric == "n" else f"{bucket}_fumbles_lost"
        for metric, bucket in wide.columns
    ]
    return wide.reset_index()


def _two_point_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    conv = pbp[(pbp["two_point_attempt"] == 1) & (pbp["two_point_conv_result"] == "success")]
    if conv.empty:
        return pd.DataFrame(columns=KEY)

    parts = []
    for player_col, out_col in (
        ("passer_player_id", "passing_2pt_conversions"),
        ("rusher_player_id", "rushing_2pt_conversions"),
        ("receiver_player_id", "receiving_2pt_conversions"),
    ):
        part = _sum_by(conv, player_col, {out_col: ("two_point_attempt", "sum")})
        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame(columns=KEY)

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=KEY, how="outer")
    return merged


def _special_teams_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    returns = pbp[pbp["return_touchdown"] == 1]
    if returns.empty:
        return pd.DataFrame(columns=KEY)

    parts = []
    for player_col in ("punt_returner_player_id", "kickoff_returner_player_id"):
        part = _sum_by(returns, player_col, {"special_teams_tds": ("return_touchdown", "sum")})
        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame(columns=KEY)
    return pd.concat(parts).groupby(KEY, as_index=False)["special_teams_tds"].sum()


def _player_team_map(pbp: pd.DataFrame) -> pd.DataFrame:
    """Assign each player-week the team whose plays they appeared in most."""
    frames = []
    for col in ("passer_player_id", "receiver_player_id", "rusher_player_id",
                "punt_returner_player_id", "kickoff_returner_player_id"):
        part = pbp.dropna(subset=[col, "posteam"])[[col, "season", "week", "posteam"]]
        frames.append(part.rename(columns={col: "player_id"}))

    appearances = pd.concat(frames, ignore_index=True)
    counts = appearances.groupby(KEY + ["posteam"]).size().reset_index(name="plays")
    counts = counts.sort_values(KEY + ["plays"], ascending=[True, True, True, False])
    top = counts.drop_duplicates(subset=KEY, keep="first")
    return top.drop(columns="plays").rename(columns={"posteam": "box_team"})


def _add_share_and_efficiency(stats: pd.DataFrame) -> pd.DataFrame:
    team_totals = stats.groupby(["box_team", "season", "week"]).agg(
        team_targets=("targets", "sum"),
        team_air_yards=("receiving_air_yards", "sum"),
    ).reset_index()
    stats = stats.merge(team_totals, on=["box_team", "season", "week"], how="left")

    def ratio(numer, denom):
        return numer / denom.replace(0, np.nan)

    stats["target_share"] = ratio(stats["targets"], stats["team_targets"])
    stats["air_yards_share"] = ratio(stats["receiving_air_yards"], stats["team_air_yards"])
    stats["wopr"] = 1.5 * stats["target_share"].fillna(0) + 0.7 * stats["air_yards_share"].fillna(0)
    stats["racr"] = ratio(stats["receiving_yards"], stats["receiving_air_yards"])
    stats["pacr"] = ratio(stats["passing_yards"], stats["passing_air_yards"])
    # dakota is an EPA+CPOE composite nflverse fits separately; not reproducible here.
    stats["dakota"] = np.nan
    return stats.drop(columns=["team_targets", "team_air_yards"])


def _score(stats: pd.DataFrame) -> pd.DataFrame:
    fumbles_lost = (
        stats["sack_fumbles_lost"] + stats["rushing_fumbles_lost"] + stats["receiving_fumbles_lost"]
    )
    two_pt = (
        stats["passing_2pt_conversions"] + stats["rushing_2pt_conversions"]
        + stats["receiving_2pt_conversions"]
    )

    base = sum(stats[col] * weight for col, weight in PPR_WEIGHTS.items() if col != "receptions")
    base = base + fumbles_lost * FUMBLE_LOST_POINTS + two_pt * TWO_POINT_POINTS

    stats["fantasy_points"] = base
    stats["fantasy_points_ppr"] = base + stats["receptions"] * PPR_WEIGHTS["receptions"]
    return stats


# Every stat column this module produces, so callers get a stable schema.
COUNT_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "sacks", "sack_yards", "sack_fumbles", "sack_fumbles_lost", "passing_air_yards",
    "passing_yards_after_catch", "passing_first_downs", "passing_epa",
    "passing_2pt_conversions", "carries", "rushing_yards", "rushing_tds",
    "rushing_fumbles", "rushing_fumbles_lost", "rushing_first_downs", "rushing_epa",
    "rushing_2pt_conversions", "receptions", "targets", "receiving_yards",
    "receiving_tds", "receiving_fumbles", "receiving_fumbles_lost",
    "receiving_air_yards", "receiving_yards_after_catch", "receiving_first_downs",
    "receiving_epa", "receiving_2pt_conversions", "special_teams_tds",
]
DERIVED_COLS = ["target_share", "air_yards_share", "wopr", "racr", "pacr", "dakota",
                "fantasy_points", "fantasy_points_ppr"]


def build_pbp_box_scores(year: int, pbp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rebuild the player-week box score line for one season from play-by-play."""
    if pbp is None:
        path = RAW_DIR / "play_by_play" / f"play_by_play_{year}.csv"
        if not path.exists():
            return pd.DataFrame()
        pbp = pd.read_csv(path, usecols=PBP_USECOLS, low_memory=False)

    # Box scores cover the regular season only; match that.
    pbp = pbp[pbp["season_type"] == "REG"].copy()
    if pbp.empty:
        return pd.DataFrame()

    stats = _player_team_map(pbp)
    for part in (_passing_stats(pbp), _receiving_stats(pbp), _rushing_stats(pbp),
                 _fumble_stats(pbp), _two_point_stats(pbp), _special_teams_stats(pbp)):
        if not part.empty:
            stats = stats.merge(part, on=KEY, how="left")

    for col in COUNT_COLS:
        if col not in stats.columns:
            stats[col] = 0.0
        # EPA sums are genuinely absent for players with no such plays, but every
        # count stat means zero when the player had no qualifying play.
        stats[col] = stats[col].fillna(0.0).astype("float64")

    stats = _add_share_and_efficiency(stats)
    stats = _score(stats)
    stats["season"] = stats["season"].astype("float64")
    stats["week"] = stats["week"].astype("float64")
    return stats[KEY + ["box_team"] + COUNT_COLS + DERIVED_COLS]


def validate(year: int) -> None:
    """Compare the derived stat line against the published box scores for a season."""
    box_path = RAW_DIR / "box_scores" / f"box_scores_{year}.csv"
    if not box_path.exists():
        raise SystemExit(f"No published box scores for {year}; nothing to validate against.")

    derived = build_pbp_box_scores(year)
    box = pd.read_csv(box_path, low_memory=False)
    box = box[box["season_type"] == "REG"]

    merged = box.merge(derived, on=KEY, how="outer", suffixes=("_box", "_pbp"), indicator=True)

    print("=" * 78)
    print(f"PBP-DERIVED vs PUBLISHED BOX SCORES - {year}")
    print("=" * 78)
    print(f"  player-weeks in box scores only: {(merged['_merge'] == 'left_only').sum():>6}")
    print(f"  player-weeks in PBP only:        {(merged['_merge'] == 'right_only').sum():>6}")
    print(f"  matched player-weeks:            {(merged['_merge'] == 'both').sum():>6}")

    both = merged[merged["_merge"] == "both"]
    compare = [c for c in COUNT_COLS + ["fantasy_points", "fantasy_points_ppr"]
               if f"{c}_box" in both.columns and f"{c}_pbp" in both.columns]

    print(f"\n  {'column':<32} {'mean abs err':>13} {'exact':>8} {'max err':>10}")
    print("  " + "-" * 66)
    worst = []
    for col in compare:
        a = both[f"{col}_box"].fillna(0.0)
        b = both[f"{col}_pbp"].fillna(0.0)
        err = (a - b).abs()
        exact = float((err < 1e-6).mean())
        print(f"  {col:<32} {err.mean():>13.4f} {exact:>7.1%} {err.max():>10.2f}")
        worst.append((exact, col))

    print("\n  Columns matching exactly on under 95% of rows:")
    off = sorted(w for w in worst if w[0] < 0.95)
    if off:
        for exact, col in off:
            print(f"    {col:<32} {exact:.1%}")
    else:
        print("    (none)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=int, metavar="YEAR",
                        help="compare derived stats against published box scores for YEAR")
    args = parser.parse_args()
    if args.validate:
        validate(args.validate)
    else:
        parser.print_help()
