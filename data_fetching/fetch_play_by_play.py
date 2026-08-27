from pathlib import Path

import nfl_data_py as nfl

START_YEAR = 2010
END_YEAR = 2025
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'play_by_play'


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(START_YEAR, END_YEAR + 1):
        data = nfl.import_pbp_data([year])

        if data is None:
            print(f"No data returned for {year}; skipping.")
            continue

        output_path = OUTPUT_DIR / f"play_by_play_{year}.csv"
        data.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


if __name__ == '__main__':
    main()
