"""Fetch weekly player box scores.

nflverse retired the release `nfl_data_py.import_weekly_data` reads from, so that
call now 404s for 2025 onward. The same player-week stat line is still published,
under a wider schema with a few renamed columns, via `nflreadpy.load_player_stats`.
This script tries the legacy reader first and falls back to nflreadpy, mapping the
new names back onto the legacy schema so every box_scores_<year>.csv stays
column-compatible with the ones already on disk (and with Layer 1's BOX_STAT_COLS).
"""

from pathlib import Path

import nfl_data_py as nfl
import nflreadpy as nflread
import pandas as pd

from season_args import seasons

START_YEAR = 2010
END_YEAR = 2026
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'box_scores'

# nflverse renamed these between the two feeds; everything else already matches.
COLUMN_RENAMES = {
    "team": "recent_team",
    "passing_interceptions": "interceptions",
    "sacks_suffered": "sacks",
    "sack_yards_lost": "sack_yards",
}

# The legacy feed's columns, in order, so the fallback writes the same shape.
LEGACY_COLUMNS = [
    "player_id", "player_name", "player_display_name", "position", "position_group",
    "headshot_url", "recent_team", "season", "week", "season_type", "opponent_team",
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

# The legacy feed only emitted a row when a player touched the ball on offense;
# nflreadpy emits one for every player with any stat at all, defenders and
# kickers included. Filtering on offensive involvement reproduces the legacy row
# set (5,339 of 5,340 player-weeks in 2024).
OFFENSIVE_ACTIVITY_COLS = [
    "attempts", "carries", "targets", "sacks_suffered", "special_teams_tds",
    "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
]


def from_legacy_feed(year: int) -> pd.DataFrame | None:
    try:
        return nfl.import_weekly_data([year])
    except Exception as e:
        print(f"  import_weekly_data([{year}]) unavailable ({type(e).__name__}); trying nflreadpy.")
        return None


def from_nflreadpy(year: int) -> pd.DataFrame | None:
    try:
        data = nflread.load_player_stats([year]).to_pandas()
    except Exception as e:
        print(f"  load_player_stats([{year}]) failed: {type(e).__name__}: {e}")
        return None

    if data is None or data.empty:
        return None

    activity = [c for c in OFFENSIVE_ACTIVITY_COLS if c in data.columns]
    data = data[data[activity].fillna(0.0).abs().sum(axis=1) > 0]

    data = data.rename(columns=COLUMN_RENAMES)

    # `dakota` is the one legacy column nflverse dropped; keep the slot so the
    # header matches, and let Layer 1 treat it as missing rather than zero.
    for column in LEGACY_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    return data[LEGACY_COLUMNS]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in seasons(START_YEAR, END_YEAR, __doc__):
        data = from_legacy_feed(year)
        source = "import_weekly_data"

        if data is None or data.empty:
            data = from_nflreadpy(year)
            source = "nflreadpy.load_player_stats"

        if data is None or data.empty:
            print(f"No data returned for {year}; skipping.")
            continue

        output_path = OUTPUT_DIR / f"box_scores_{year}.csv"
        data.to_csv(output_path, index=False)
        weeks = sorted(data["week"].dropna().unique().tolist())
        print(f"Saved {output_path} ({len(data):,} rows, weeks {int(weeks[0])}-{int(weeks[-1])}, "
              f"via {source})")


if __name__ == '__main__':
    main()
