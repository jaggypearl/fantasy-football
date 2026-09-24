"""Fetch nflverse scoring lines (the `vegas_team_line` feature).

Heads up: nflverse stopped publishing this release after 2020. `import_sc_lines`
still resolves for any season but returns zero rows from 2021 on, so 2021-2026
have no data here and `vegas_team_line` is NaN for those seasons. The spread and
total the model actually leans on come from the schedules feed
(`spread_line` / `total_line`), which is current through 2026.
"""

from pathlib import Path

import nfl_data_py as nfl

from season_args import seasons

START_YEAR = 2010
END_YEAR = 2026
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'vegas_lines'


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in seasons(START_YEAR, END_YEAR, __doc__):
        data = nfl.import_sc_lines([year])

        if data is None or data.empty:
            # Writing a header-only file would look like a successful pull and
            # make Layer 1 report an empty season rather than a missing one.
            print(f"No scoring lines published for {year}; nothing written "
                  f"(nflverse retired this release after 2020).")
            continue

        output_path = OUTPUT_DIR / f"vegas_lines_{year}.csv"
        data.to_csv(output_path, index=False)
        print(f"Saved {output_path} ({len(data):,} rows)")


if __name__ == '__main__':
    main()
