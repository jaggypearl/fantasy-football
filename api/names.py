"""Resolve a typed player name to a player_id.

player_id is the reliable key; names are a convenience and differ between
sources ("Kenneth" / "Kenny" Gainwell, "CJ" / "C.J." Stroud, "Marvin Harrison"
/ "Marvin Harrison Jr."). Matching runs in tiers and stops at the first tier
that finds anyone:

  1. exact     the normalized full name (case, accents, punctuation and
               generational suffixes ignored)
  2. nickname  same last name, and the first names share their first three
               letters (Kenneth / Kenny, Christopher / Chris)
  3. last name the query is a single word matching a last name ("Achane")

One player in the winning tier resolves; several are an ambiguity the caller
must settle by player_id, position or team. Nobody in any tier is a miss, with
the closest spellings offered as suggestions.
"""

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
NICKNAME_PREFIX = 3


def normalize(name: str) -> str:
    """'Amon-Ra St. Brown Jr.' -> 'amonra st brown'; 'Audric Estimé' -> 'audric estime'."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[.'’`]", "", text)        # C.J. -> cj, Ja'Marr -> jamarr
    text = re.sub(r"[^a-z0-9]+", " ", text)   # hyphens and the rest split words
    words = text.split()
    while len(words) > 1 and words[-1] in SUFFIXES:
        words.pop()
    return " ".join(words)


@dataclass
class Resolution:
    matches: list[dict] = field(default_factory=list)
    matched_by: str | None = None
    suggestions: list[dict] = field(default_factory=list)


def resolve(query: str, players: list[dict], position: str | None = None,
            team: str | None = None) -> Resolution:
    """Find `query` among `players` (dicts with player_name, position, team)."""
    pool = [p for p in players
            if (position is None or p["position"] == position) and (team is None or p["team"] == team)]
    target = normalize(query)
    if not target:
        return Resolution()
    words = target.split()
    keyed = [(normalize(p["player_name"]).split(), p) for p in pool]

    tiers = [("exact", [p for name, p in keyed if name == words])]
    if len(words) >= 2:
        first, last = words[0], words[-1]
        tiers.append(("nickname", [
            p for name, p in keyed
            if name[-1] == last and len(name) >= 2
            and name[0][:NICKNAME_PREFIX] == first[:NICKNAME_PREFIX]]))
    else:
        tiers.append(("last_name", [p for name, p in keyed if name[-1] == words[0]]))

    for matched_by, matches in tiers:
        if matches:
            return Resolution(matches=matches, matched_by=matched_by)

    by_name = {normalize(p["player_name"]): p for p in pool}
    close = difflib.get_close_matches(target, list(by_name), n=5, cutoff=0.6)
    return Resolution(suggestions=[by_name[name] for name in close])
