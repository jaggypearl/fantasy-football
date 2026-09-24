"""Shared --years argument for the fetch scripts.

Every fetcher defaults to its full START_YEAR..END_YEAR range, but re-downloading
sixteen seasons to refresh one is slow (play-by-play alone is ~140MB a season), so
each script accepts a subset:

    python data_fetching/fetch_play_by_play.py --years 2026
    python data_fetching/fetch_ngs.py --years 2024-2026
"""

import argparse


def parse_year_list(text: str) -> list[int]:
    years: list[int] = []
    for segment in text.split(","):
        segment = segment.strip()
        if not segment:
            continue
        if "-" in segment.lstrip("-"):
            start_str, end_str = segment.split("-", 1)
            years.extend(range(int(start_str), int(end_str) + 1))
        else:
            years.append(int(segment))
    return sorted(set(years))


def seasons(default_start: int, default_end: int, description: str | None = None) -> list[int]:
    """Return the seasons to fetch, honouring --years if the caller passed it."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--years",
        metavar="SPEC",
        help="seasons to fetch, e.g. '2026', '2024-2026' or '2019,2026' "
             f"(default: {default_start}-{default_end})",
    )
    args = parser.parse_args()

    if not args.years:
        return list(range(default_start, default_end + 1))
    return parse_year_list(args.years)
