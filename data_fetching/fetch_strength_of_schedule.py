from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from season_args import seasons

START_YEAR = 2010
END_YEAR = 2026
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "strength_of_schedule"


def compute_win_pct(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules[schedules["game_type"] == "REG"].dropna(subset=["home_score", "away_score"])

    home = reg[["season", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "team_score", "away_score": "opp_score"}
    )
    away = reg[["season", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "team_score", "home_score": "opp_score"}
    )
    games = pd.concat([home, away], ignore_index=True)

    games["win"] = (games["team_score"] > games["opp_score"]).astype(float)
    games["tie"] = (games["team_score"] == games["opp_score"]).astype(float)

    grouped = games.groupby(["season", "team"]).agg(win=("win", "sum"), tie=("tie", "sum"), games=("team", "count"))
    grouped["win_pct"] = (grouped["win"] + 0.5 * grouped["tie"]) / grouped["games"]

    return grouped.reset_index()[["season", "team", "win_pct"]]


def build_matchups(schedules: pd.DataFrame) -> pd.DataFrame:
    reg = schedules[schedules["game_type"] == "REG"]

    home = reg[["season", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opponent"})
    away = reg[["season", "away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opponent"})

    return pd.concat([home, away], ignore_index=True)


def main() -> None:
    # Opponent win % needs every season loaded, so --years only picks which
    # season files get rewritten.
    years = seasons(START_YEAR, END_YEAR, __doc__)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    schedules = nfl.import_schedules(list(range(START_YEAR - 1, END_YEAR + 1)))

    if schedules is None:
        print("No schedule data returned; aborting.")
        return

    win_pct = compute_win_pct(schedules)
    matchups = build_matchups(schedules[schedules["season"] >= START_YEAR])

    matchups = matchups.merge(
        win_pct.rename(columns={"win_pct": "adjusted_win_pct"}),
        left_on=["season", "opponent"],
        right_on=["season", "team"],
        suffixes=("", "_opp"),
        how="left",
    ).drop(columns=["team_opp"])

    prior_win_pct = win_pct.copy()
    prior_win_pct["season"] = prior_win_pct["season"] + 1
    matchups = matchups.merge(
        prior_win_pct.rename(columns={"win_pct": "preseason_win_pct"}),
        left_on=["season", "opponent"],
        right_on=["season", "team"],
        suffixes=("", "_prioropp"),
        how="left",
    ).drop(columns=["team_prioropp"])

    sos = matchups.groupby(["season", "team"]).agg(
        adjusted_sos=("adjusted_win_pct", "mean"),
        preseason_sos=("preseason_win_pct", "mean"),
    ).reset_index()

    for year in years:
        year_data = sos[sos["season"] == year][["season", "team", "preseason_sos", "adjusted_sos"]]

        if year_data.empty:
            print(f"No data returned for {year}; skipping.")
            continue

        output_path = OUTPUT_DIR / f"strength_of_schedule_{year}.csv"
        year_data.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
