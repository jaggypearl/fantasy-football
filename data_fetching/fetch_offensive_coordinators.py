import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

START_YEAR = 2010
END_YEAR = 2025
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "offensive_coordinators"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

FRANCHISES = [
    (1, "san-francisco-49ers", "SF"),
    (2, "cincinnati-bengals", "CIN"),
    (3, "new-york-jets", "NYJ"),
    (4, "new-york-giants", "NYG"),
    (6, "new-england-patriots", "NE"),
    (7, "buffalo-bills", "BUF"),
    (9, "kansas-city-chiefs", "KC"),
    (10, "denver-broncos", "DEN"),
    (13, "tennessee-titans", "TEN"),
    (15, "los-angeles-chargers", "LAC"),
    (17, "las-vegas-raiders", "LV"),
    (19, "miami-dolphins", "MIA"),
    (20, "dallas-cowboys", "DAL"),
    (21, "minnesota-vikings", "MIN"),
    (22, "atlanta-falcons", "ATL"),
    (23, "new-orleans-saints", "NO"),
    (24, "seattle-seahawks", "SEA"),
    (25, "tampa-bay-buccaneers", "TB"),
    (26, "carolina-panthers", "CAR"),
    (27, "jacksonville-jaguars", "JAX"),
    (28, "baltimore-ravens", "BAL"),
    (29, "houston-texans", "HOU"),
    (31, "indianapolis-colts", "IND"),
    (32, "cleveland-browns", "CLE"),
    (35, "los-angeles-rams", "LA"),
    (37, "pittsburgh-steelers", "PIT"),
    (40, "washington-commanders", "WAS"),
    (42, "detroit-lions", "DET"),
    (43, "green-bay-packers", "GB"),
    (46, "chicago-bears", "CHI"),
    (50, "arizona-cardinals", "ARI"),
    (114, "philadelphia-eagles", "PHI"),
]

RELOCATIONS = {
    "LV": [(2020, END_YEAR, "LV"), (START_YEAR, 2019, "OAK")],
    "LAC": [(2017, END_YEAR, "LAC"), (START_YEAR, 2016, "SD")],
    "LA": [(2016, END_YEAR, "LA"), (START_YEAR, 2015, "STL")],
}


def team_abbr(default_abbr: str, season: int) -> str:
    for start, end, abbr in RELOCATIONS.get(default_abbr, []):
        if start <= season <= end:
            return abbr
    return default_abbr


def parse_years(years_text: str) -> list[int]:
    years_text = years_text.replace("–", "-")
    years = []
    for segment in years_text.split(","):
        segment = segment.strip()
        if "-" in segment:
            start_str, end_str = segment.split("-")
            years.extend(range(int(start_str), int(end_str) + 1))
        else:
            years.append(int(segment))
    return years


def fetch_franchise_ocs(franchise_id: int, slug: str) -> pd.DataFrame:
    url = f"https://pro-football-history.com/franchpos/{franchise_id}/7/{slug}-offensive-coordinator-history"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table")
    if table is None:
        return pd.DataFrame(columns=["season", "offensive_coordinator"])

    records = []
    for row in table.find_all("tr")[1:]:
        cells = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
        if len(cells) < 2:
            continue
        coach, years_text = cells[0], cells[1]
        for season in parse_years(years_text):
            records.append({"season": season, "offensive_coordinator": coach})

    return pd.DataFrame(records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for franchise_id, slug, default_abbr in FRANCHISES:
        try:
            data = fetch_franchise_ocs(franchise_id, slug)
        except Exception as e:
            print(f"Error fetching {slug}: {type(e).__name__}: {e}; skipping.")
            time.sleep(2)
            continue

        if data is None or data.empty:
            print(f"No data returned for {slug}; skipping.")
            time.sleep(2)
            continue

        data = data[(data["season"] >= START_YEAR) & (data["season"] <= END_YEAR)].copy()
        data["team"] = data["season"].apply(lambda season: team_abbr(default_abbr, season))
        all_rows.append(data[["season", "team", "offensive_coordinator"]])

        time.sleep(2)

    combined = pd.concat(all_rows, ignore_index=True)

    for year in range(START_YEAR, END_YEAR + 1):
        year_data = combined[combined["season"] == year]

        if year_data.empty:
            print(f"No data returned for {year}; skipping.")
            continue

        output_path = OUTPUT_DIR / f"offensive_coordinators_{year}.csv"
        year_data.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
