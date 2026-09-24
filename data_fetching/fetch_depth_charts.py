from pathlib import Path

import nflreadpy as nfl
import pandas as pd

from season_args import seasons

START_YEAR = 2010
END_YEAR = 2026

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "raw" / "depth_chart"
SCHEDULE_DIR = PROJECT_DIR / "data" / "raw" / "schedules"

OUTPUT_COLUMNS = ["season", "week", "team", "player_id", "player_name",
                  "position", "depth_chart_rank", "depth_chart_position"]

TEAM_ALIASES = {"LAR": "LA", "STL": "LA", "OAK": "LV", "SD": "LAC", "WSH": "WAS",
                "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA"}

DEPTH_SCORES = {1: 1.0, 2: 0.5, 3: 0.2}
DEPTH_SCORE_DEEP = 0.1
DEPTH_SCORE_UNKNOWN = 0.7


def normalize_team(value) -> str | None:
    if pd.isna(value):
        return None
    team = str(value).strip().upper()
    return TEAM_ALIASES.get(team, team)


def encode_depth(rank) -> float:
    if pd.isna(rank):
        return DEPTH_SCORE_UNKNOWN
    return DEPTH_SCORES.get(int(rank), DEPTH_SCORE_DEEP)


def week_start_dates(season: int) -> pd.Series:
    path = SCHEDULE_DIR / f"schedules_{season}.csv"
    if not path.exists():
        return pd.Series(dtype="datetime64[ns]")
    sched = pd.read_csv(path, low_memory=False)
    sched = sched[sched["game_type"] == "REG"]
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    starts = sched.groupby("week")["gameday"].min().dropna().sort_index()
    return starts


def from_legacy(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    df = raw.copy()
    name = df.get("full_name")
    if name is None:
        name = df["football_name"].fillna("") + " " + df["last_name"].fillna("")
    out = pd.DataFrame({
        "season": season,
        "week": pd.to_numeric(df["week"], errors="coerce"),
        "team": df["club_code"].map(normalize_team),
        "player_id": df["gsis_id"],
        "player_name": name,
        "position": df["position"].astype("string").str.upper(),
        "depth_chart_rank": pd.to_numeric(df["depth_team"], errors="coerce"),
    })
    out = out[out["week"].notna() & out["player_id"].notna()]
    out = out.sort_values("depth_chart_rank").drop_duplicates(
        subset=["season", "week", "player_id", "position"], keep="first")
    return out


def from_snapshots(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    df = raw.copy()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df[df["dt"].notna() & df["gsis_id"].notna()]

    starts = week_start_dates(season)
    if starts.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    horizon = df["dt"].max()
    frames = []
    for week, start in starts.items():
        if start > horizon:
            continue
        eligible = df[df["dt"] <= start]
        if eligible.empty:
            continue
        latest = eligible["dt"].max()
        snap = eligible[eligible["dt"] == latest].copy()
        snap["week"] = float(week)
        frames.append(snap)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    snaps = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "season": season,
        "week": snaps["week"],
        "team": snaps["team"].map(normalize_team),
        "player_id": snaps["gsis_id"],
        "player_name": snaps["player_name"],
        "position": snaps["pos_abb"].astype("string").str.upper(),
        "depth_chart_rank": pd.to_numeric(snaps["pos_rank"], errors="coerce"),
    })
    out = out.sort_values("depth_chart_rank").drop_duplicates(
        subset=["season", "week", "player_id", "position"], keep="first")
    return out


def fetch_season(season: int) -> pd.DataFrame:
    raw = nfl.load_depth_charts([season]).to_pandas()
    if raw.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if "depth_team" in raw.columns:
        out = from_legacy(raw, season)
    else:
        out = from_snapshots(raw, season)
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out["depth_chart_position"] = out["depth_chart_rank"].map(encode_depth)
    return out[OUTPUT_COLUMNS].sort_values(["week", "team", "position", "depth_chart_rank"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons(START_YEAR, END_YEAR, __doc__):
        try:
            data = fetch_season(season)
        except Exception as exc:
            print(f"{season}: fetch failed ({type(exc).__name__}: {exc}); skipping.")
            continue

        if data.empty:
            print(f"{season}: no depth chart rows returned; skipping.")
            continue

        output_path = OUTPUT_DIR / f"depth_chart_{season}.csv"
        data.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved {output_path} ({len(data):,} rows, weeks "
              f"{int(data['week'].min())}-{int(data['week'].max())})")


if __name__ == "__main__":
    main()
