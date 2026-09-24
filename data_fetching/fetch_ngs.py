from pathlib import Path

import nfl_data_py as nfl

from season_args import seasons


START_YEAR = 2016
END_YEAR = 2026
STAT_TYPES = ("passing", "rushing", "receiving")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ngs"


def main() -> None:
    years = seasons(START_YEAR, END_YEAR, __doc__)

    for stat_type in STAT_TYPES:
        stat_output_dir = OUTPUT_DIR / stat_type
        stat_output_dir.mkdir(parents=True, exist_ok=True)

        for year in years:
            data = nfl.import_ngs_data(stat_type, [year])

            if data is None:
                print(f"No data returned for {stat_type} {year}; skipping.")
                continue

            output_path = stat_output_dir / f"ngs_{stat_type}_{year}.csv"
            data.to_csv(output_path, index=False)
            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
