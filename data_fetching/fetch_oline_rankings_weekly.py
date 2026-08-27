from pathlib import Path

import pandas as pd

START_YEAR = 2010
END_YEAR = 2025
INPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "play_by_play"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "oline_rankings_weekly"
USE_COLS = [
    "season",
    "week",
    "posteam",
    "play_type",
    "sack",
    "qb_hit",
    "pass_attempt",
    "rush_attempt",
    "rushing_yards",
]


def compute_oline_rankings(pbp: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp.dropna(subset=["posteam"])

    pass_plays = pbp[pbp["pass_attempt"] == 1]
    pass_stats = pass_plays.groupby(["season", "week", "posteam"]).agg(
        pass_attempts=("pass_attempt", "sum"),
        sacks=("sack", "sum"),
        qb_hits=("qb_hit", "sum"),
    )

    rush_plays = pbp[pbp["rush_attempt"] == 1]
    rush_stats = rush_plays.groupby(["season", "week", "posteam"]).agg(
        rush_attempts=("rush_attempt", "sum"),
        rush_yards=("rushing_yards", "sum"),
        stuffed_runs=("rushing_yards", lambda yards: (yards <= 0).sum()),
    )

    stats = pass_stats.join(rush_stats, how="outer").reset_index()
    stats = stats[(stats["pass_attempts"] > 0) & (stats["rush_attempts"] > 0)]

    stats["sack_rate"] = stats["sacks"] / stats["pass_attempts"]
    stats["qb_hit_rate"] = stats["qb_hits"] / stats["pass_attempts"]
    stats["stuff_rate"] = stats["stuffed_runs"] / stats["rush_attempts"]
    stats["yards_per_carry"] = stats["rush_yards"] / stats["rush_attempts"]

    def zscore(series: pd.Series) -> pd.Series:
        std = series.std()
        return (series - series.mean()) / std if std else series * 0

    grouped = stats.groupby(["season", "week"])
    stats["composite"] = (
        -grouped["sack_rate"].transform(zscore)
        - grouped["qb_hit_rate"].transform(zscore)
        - grouped["stuff_rate"].transform(zscore)
        + grouped["yards_per_carry"].transform(zscore)
    )
    stats["oline_rank"] = grouped["composite"].rank(ascending=False, method="min").astype(int)

    return stats.rename(columns={"posteam": "team"})[["season", "week", "team", "oline_rank"]]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(START_YEAR, END_YEAR + 1):
        input_path = INPUT_DIR / f"play_by_play_{year}.csv"

        if not input_path.exists():
            print(f"No play-by-play data found for {year}; skipping.")
            continue

        pbp = pd.read_csv(input_path, usecols=USE_COLS, low_memory=False)
        data = compute_oline_rankings(pbp)

        if data is None or data.empty:
            print(f"No data computed for {year}; skipping.")
            continue

        output_path = OUTPUT_DIR / f"oline_rankings_weekly_{year}.csv"
        data.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
