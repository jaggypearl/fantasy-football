"use client";

import { useEffect, useId, useRef, useState } from "react";
import { ApiError, isAbort, searchPlayers, type Player } from "@/lib/api";

const DEBOUNCE_MS = 200;
const MAX_RESULTS = 8;

interface Props {
  selectedName: string | null; // shown in the input once a projection loads
  onSelect: (player: Player) => void;
  onSubmitName: (name: string) => void; // Enter with no suggestion highlighted
}

type Suggestions =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; players: Player[] }
  | { status: "error"; unavailable: boolean };

export default function PlayerSearch({ selectedName, onSelect, onSubmitName }: Props) {
  const [query, setQuery] = useState(selectedName ?? "");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const [suggestions, setSuggestions] = useState<Suggestions>({ status: "idle" });
  const listId = useId();
  const typedRef = useRef(false); // only autocomplete on user typing, not on programmatic fills

  useEffect(() => {
    if (selectedName) {
      typedRef.current = false;
      setQuery(selectedName);
      setOpen(false);
    }
  }, [selectedName]);

  useEffect(() => {
    const q = query.trim();
    if (!typedRef.current || !q) {
      setSuggestions({ status: "idle" });
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setSuggestions({ status: "loading" });
      searchPlayers(q, controller.signal).then(
        (players) => {
          setSuggestions({ status: "done", players: players.slice(0, MAX_RESULTS) });
          setActive(-1);
        },
        (error) => {
          if (isAbort(error)) return;
          const unavailable = !(error instanceof ApiError) || error.kind === "unavailable";
          setSuggestions({ status: "error", unavailable });
        },
      );
    }, DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  const players = suggestions.status === "done" ? suggestions.players : [];
  const showList = open && suggestions.status !== "idle";

  function choose(player: Player) {
    typedRef.current = false;
    setQuery(player.player_name);
    setOpen(false);
    onSelect(player);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (!players.length) return;
      e.preventDefault();
      setOpen(true);
      const last = players.length - 1;
      setActive((i) => (e.key === "ArrowDown" ? (i >= last ? 0 : i + 1) : i <= 0 ? last : i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (showList && active >= 0 && players[active]) choose(players[active]);
      else if (query.trim()) {
        setOpen(false);
        onSubmitName(query.trim());
      }
    } else if (e.key === "Escape") {
      setOpen(false);
      setActive(-1);
    }
  }

  return (
    <div className="search">
      <label htmlFor="player-search" className="visually-hidden">
        Search players
      </label>
      <input
        id="player-search"
        className="search-input"
        type="text"
        role="combobox"
        aria-expanded={showList}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={showList && active >= 0 ? `${listId}-${active}` : undefined}
        autoComplete="off"
        spellCheck={false}
        placeholder="Search a player, e.g. Josh Allen"
        value={query}
        onChange={(e) => {
          typedRef.current = true;
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={onKeyDown}
      />
      {showList && (
        <ul id={listId} className="search-dropdown" role="listbox" aria-label="Matching players">
          {suggestions.status === "loading" && <li className="search-note">Searching…</li>}
          {suggestions.status === "error" && (
            <li className="search-note">
              {suggestions.unavailable ? "The server is waking up, try again in a few seconds" : "Search failed"}
            </li>
          )}
          {suggestions.status === "done" && players.length === 0 && (
            <li className="search-note">No players match</li>
          )}
          {players.map((player, i) => (
            <li
              key={player.player_id}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              className="search-option"
              // mousedown, not click, so the input's blur doesn't close the list first
              onMouseDown={(e) => {
                e.preventDefault();
                choose(player);
              }}
              onMouseEnter={() => setActive(i)}
            >
              <span>{player.player_name}</span>
              <span className="search-option-meta">
                {player.position} · {player.team}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
