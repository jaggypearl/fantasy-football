"""Build the 2026 offensive-coordinator / team-tendency file.

nflverse publishes no coaching-staff endpoint, so coordinator *names* are always
carried forward from 2025. The tendency columns are a different story: once 2026
games have been played, they are computed from the real 2026 play-by-play rather
than carried forward, and the `source` column records which of the two happened.
"""

from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

TARGET_SEASON = 2026
FALLBACK_SEASON = 2025

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = RAW_DIR / "offensive_coordinators"
OUTPUT_PATH = OUTPUT_DIR / f"offensive_coordinators_{TARGET_SEASON}.csv"
OC_FALLBACK_PATH = OUTPUT_DIR / f"offensive_coordinators_{FALLBACK_SEASON}.csv"
PBP_TARGET_PATH = RAW_DIR / "play_by_play" / f"play_by_play_{TARGET_SEASON}.csv"
PBP_FALLBACK_PATH = RAW_DIR / "play_by_play" / f"play_by_play_{FALLBACK_SEASON}.csv"
BIO_TARGET_PATH = RAW_DIR / "bio_data" / f"bio_data_{TARGET_SEASON}.csv"
BIO_FALLBACK_PATH = RAW_DIR / "bio_data" / f"bio_data_{FALLBACK_SEASON}.csv"

COACH_KEYWORDS = ("oc", "offensive", "coach")

OUTPUT_COLUMNS = [
    "season",
    "team",
    "offensive_coordinator",
    "pass_play_pct",
    "rush_play_pct",
    "team_pass_yards_per_game",
    "team_rush_yards_per_game",
    "team_target_share_to_wr",
    "team_target_share_to_te",
    "team_target_share_to_rb",
    "team_rush_share_to_rb",
    "source",
]


def fetch_nflverse_coaching() -> pd.DataFrame | None:
    importer = getattr(nfl, "import_team_coaching", None)
    if importer is None:
        print(f"  nfl_data_py {nfl.__version__ if hasattr(nfl, '__version__') else ''} has no "
              f"import_team_coaching(); nflverse exposes no coaching-staff endpoint.")
        return None

    try:
        data = importer([TARGET_SEASON])
    except Exception as e:
        print(f"  import_team_coaching([{TARGET_SEASON}]) failed: {type(e).__name__}: {e}")
        return None

    if data is None or data.empty:
        return None

    if "season" in data.columns:
        data = data[data["season"] == TARGET_SEASON]
    if data.empty:
        return None

    keep = [c for c in data.columns if c in ("season", "team", "team_abbr")]
    keep += [c for c in data.columns if any(k in c.lower() for k in COACH_KEYWORDS) and c not in keep]
    keep += [c for c in data.columns if ("pass" in c.lower() or "rush" in c.lower()) and c not in keep]
    return data[keep]


def load_carryover_coordinators() -> pd.DataFrame:
    if not OC_FALLBACK_PATH.exists():
        print(f"  [warn] {OC_FALLBACK_PATH.name} not found; oc_name will be blank.")
        return pd.DataFrame(columns=["team", "offensive_coordinator"])

    oc = pd.read_csv(OC_FALLBACK_PATH)
    oc = oc[oc["season"] == FALLBACK_SEASON][["team", "offensive_coordinator"]]
    return oc.drop_duplicates(subset=["team"])


def tendency_source() -> tuple[Path | None, int | None]:
    """Prefer real target-season plays; fall back to the prior season."""
    if PBP_TARGET_PATH.exists():
        return PBP_TARGET_PATH, TARGET_SEASON
    if PBP_FALLBACK_PATH.exists():
        return PBP_FALLBACK_PATH, FALLBACK_SEASON
    return None, None


def load_positions(season: int) -> dict[str, str]:
    path = BIO_TARGET_PATH if season == TARGET_SEASON else BIO_FALLBACK_PATH
    if not path.exists():
        print(f"  [warn] {path.name} not found; per-position shares will be blank.")
        return {}

    bio = pd.read_csv(path, low_memory=False)
    id_col = next((c for c in ("player_id", "gsis_id") if c in bio.columns), None)
    if id_col is None or "position" not in bio.columns:
        print(f"  [warn] {path.name} lacks player id or position; shares will be blank.")
        return {}

    bio = bio.dropna(subset=[id_col, "position"]).drop_duplicates(subset=[id_col])
    return dict(zip(bio[id_col], bio["position"]))


def compute_tendencies(path: Path | None, season: int | None) -> tuple[pd.DataFrame, str]:
    if path is None:
        print("  [warn] no play-by-play found; tendency columns will be blank.")
        return pd.DataFrame(columns=["team"]), "unavailable"

    cols = [
        "game_id", "season_type", "posteam", "play_type", "pass_attempt", "rush_attempt",
        "passing_yards", "rushing_yards", "receiver_player_id", "rusher_player_id",
    ]
    pbp = pd.read_csv(path, usecols=cols + ["week"], low_memory=False)
    pbp = pbp[(pbp["season_type"] == "REG") & pbp["posteam"].notna()]

    if pbp.empty:
        print(f"  [warn] {path.name} has no regular-season plays; tendency columns will be blank.")
        return pd.DataFrame(columns=["team"]), "unavailable"

    weeks = sorted(int(w) for w in pbp["week"].dropna().unique())
    span = f"weeks {weeks[0]}-{weeks[-1]}"
    label = (f"{season}_actual_{span.replace(' ', '')}" if season == TARGET_SEASON
             else f"carried_forward_from_{season}")
    print(f"  tendencies from {path.name} ({span}, {len(pbp):,} plays)")

    plays = pbp[pbp["play_type"].isin(["pass", "run"])]
    volume = plays.groupby("posteam").agg(
        pass_plays=("play_type", lambda s: int((s == "pass").sum())),
        rush_plays=("play_type", lambda s: int((s == "run").sum())),
    )
    yards = pbp.groupby("posteam").agg(
        pass_yards=("passing_yards", "sum"),
        rush_yards=("rushing_yards", "sum"),
        games=("game_id", "nunique"),
    )

    out = volume.join(yards)
    total_plays = out["pass_plays"] + out["rush_plays"]
    out["pass_play_pct"] = out["pass_plays"] / total_plays
    out["rush_play_pct"] = out["rush_plays"] / total_plays
    out["team_pass_yards_per_game"] = out["pass_yards"] / out["games"]
    out["team_rush_yards_per_game"] = out["rush_yards"] / out["games"]

    positions = load_positions(season)
    if positions:
        targets = pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()].copy()
        targets["position"] = targets["receiver_player_id"].map(positions)
        rushes = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()].copy()
        rushes["position"] = rushes["rusher_player_id"].map(positions)

        for position, column in (("WR", "team_target_share_to_wr"), ("TE", "team_target_share_to_te"),
                                 ("RB", "team_target_share_to_rb")):
            share = targets.groupby("posteam")["position"].apply(lambda s, p=position: (s == p).sum() / len(s))
            out[column] = share

        out["team_rush_share_to_rb"] = rushes.groupby("posteam")["position"].apply(
            lambda s: (s == "RB").sum() / len(s)
        )

    return out.reset_index().rename(columns={"posteam": "team"}), label


def build_fallback() -> pd.DataFrame:
    coordinators = load_carryover_coordinators()
    path, season = tendency_source()
    tendencies, tendency_label = compute_tendencies(path, season)

    if coordinators.empty and tendencies.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    merged = tendencies.merge(coordinators, on="team", how="outer")
    merged["season"] = TARGET_SEASON
    # Coordinator names are always carried forward; the tendencies may not be.
    merged["source"] = f"oc_from_{FALLBACK_SEASON}|tendencies_{tendency_label}"

    for column in OUTPUT_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA

    return merged[OUTPUT_COLUMNS].sort_values("team").reset_index(drop=True)


def report(data: pd.DataFrame, source_label: str) -> None:
    print(f"\nSaved {OUTPUT_PATH}")
    print(f"  source:  {source_label}")
    print(f"  rows:    {len(data)}")
    print(f"  columns: {list(data.columns)}")

    if len(data) != 32:
        print(f"  [warn] expected 32 teams, got {len(data)}")

    blank = [c for c in data.columns if data[c].isna().all()]
    if blank:
        print(f"  [warn] columns with no data: {blank}")

    print("\n  sample rows:")
    preview = data.head(5)
    for line in preview.to_string(index=False).splitlines():
        print(f"    {line}")


def main() -> None:
    print("=" * 80)
    print(f"OFFENSIVE COORDINATORS + TEAM TENDENCIES - {TARGET_SEASON}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nAttempting nflverse coaching data for {TARGET_SEASON}...")
    data = fetch_nflverse_coaching()

    if data is not None and not data.empty:
        data.to_csv(OUTPUT_PATH, index=False)
        report(data, f"nflverse import_team_coaching([{TARGET_SEASON}])")
        print("=" * 80)
        return

    print(f"\n{TARGET_SEASON} coaching data not yet available from nflverse")
    print(f"Will use {FALLBACK_SEASON} OC data as fallback in prediction.py")

    print(f"\nBuilding {TARGET_SEASON} file by carrying {FALLBACK_SEASON} forward...")
    fallback = build_fallback()

    if fallback.empty:
        print(f"  [warn] no {FALLBACK_SEASON} source data found; nothing written.")
        print(f"  {OUTPUT_PATH.name} was not created. This is not an error.")
        print("=" * 80)
        return

    fallback.to_csv(OUTPUT_PATH, index=False)
    source_label = str(fallback["source"].iloc[0])
    report(fallback, source_label)

    print(f"\n  [warn] coordinator names are carried forward from {FALLBACK_SEASON}; nflverse "
          f"publishes no\n         coaching endpoint, so {TARGET_SEASON} hires are NOT reflected.")
    if f"tendencies_{TARGET_SEASON}_actual" in source_label:
        print(f"  [note] tendency columns ARE real {TARGET_SEASON} data, but off a short sample - "
              f"early-season\n         play-call splits are noisy and will settle as weeks accumulate.")
    else:
        print(f"  [warn] tendency columns are also {FALLBACK_SEASON} data labelled as {TARGET_SEASON}.")
    print("=" * 80)


if __name__ == "__main__":
    main()
