"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import PlayerSearch from "@/components/PlayerSearch";
import ProjectionCard, { ProjectionCardSkeleton } from "@/components/ProjectionCard";
import StatusMessage, { RequestError } from "@/components/StatusMessage";
import WeekSelect from "@/components/WeekSelect";
import { ApiError, getPlayerProjection, getWeeks, type Player } from "@/lib/api";
import { useRequest } from "@/lib/useRequest";
import { parseWeek } from "@/lib/params";

// The URL (?player_id= or ?name=, plus ?week=) is the source of truth, so rankings can link here.
export default function LookupView() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const playerId = params.get("player_id");
  const name = playerId ? null : params.get("name");
  const urlWeek = parseWeek(params.get("week"));

  // The name of a player picked from a list, for the input while (or if) the projection fails to load.
  const [picked, setPicked] = useState<Player | null>(null);

  const weeks = useRequest("weeks", getWeeks);
  const who = playerId ? { playerId } : name ? { name } : null;
  const projection = useRequest(who ? JSON.stringify([who, urlWeek]) : null, (signal) =>
    getPlayerProjection(who!, urlWeek ?? undefined, signal),
  );

  const weeksData = weeks.state.status === "success" ? weeks.state.data : null;
  const data = projection.state.status === "success" ? projection.state.data : null;
  const week = urlWeek ?? data?.week ?? weeksData?.default_week ?? null;

  function navigate(next: Record<string, string | number | null>, push = true) {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(next)) if (value !== null) query.set(key, String(value));
    const url = `${pathname}?${query}`;
    if (push) router.push(url, { scroll: false });
    else router.replace(url, { scroll: false });
  }

  function selectPlayer(player: Player) {
    setPicked(player);
    navigate({ player_id: player.player_id, week });
  }

  const pickedName = picked && picked.player_id === playerId ? picked.player_name : null;

  function retry() {
    if (weeks.state.status === "error") weeks.retry();
    projection.retry();
  }

  return (
    <>
      <header className="page-header">
        <h1 className="page-title">Player projections</h1>
        <p className="page-subtitle">Weekly PPR projections for every QB, RB, WR and TE.</p>
      </header>

      <div className="lookup-controls">
        <PlayerSearch
          selectedName={data?.player.player_name ?? pickedName ?? name}
          onSelect={selectPlayer}
          onSubmitName={(typed) => navigate({ name: typed, week })}
        />
        <WeekSelect
          weeks={weeksData?.weeks ?? null}
          value={week}
          onChange={(w) => navigate({ player_id: playerId, name, week: w }, false)}
        />
      </div>

      <section className="result" aria-live="polite">
        {projection.state.status === "idle" && (
          <p className="empty-state">Start typing a player&apos;s name to see their projection.</p>
        )}
        {projection.state.status === "loading" && <ProjectionCardSkeleton />}
        {data && <ProjectionCard projection={data} />}
        {projection.state.status === "error" && (
          <LookupError
            error={projection.state.error}
            week={week}
            onSelect={selectPlayer}
            onWeek={(w) => navigate({ player_id: playerId, name, week: w }, false)}
            onRetry={retry}
          />
        )}
      </section>
    </>
  );
}

interface LookupErrorProps {
  error: unknown;
  week: number | null;
  onSelect: (player: Player) => void;
  onWeek: (week: number) => void;
  onRetry: () => void;
}

function LookupError({ error, week, onSelect, onWeek, onRetry }: LookupErrorProps) {
  if (!(error instanceof ApiError)) return <RequestError error={error} onRetry={onRetry} />;

  const playerChips = (players: Player[]) =>
    players.map((p) => (
      <button key={p.player_id} type="button" className="chip" onClick={() => onSelect(p)}>
        {p.player_name}
        <span className="chip-meta">
          {p.position} · {p.team}
        </span>
      </button>
    ));

  switch (error.kind) {
    case "ambiguous":
      return (
        <StatusMessage title="Several players match that name" body="Pick the one you meant.">
          {playerChips(error.candidates)}
        </StatusMessage>
      );
    case "player_not_found":
      return (
        <StatusMessage
          title="No player found"
          body={error.suggestions.length ? "Did you mean one of these?" : "Check the spelling and try again."}
        >
          {error.suggestions.length > 0 && playerChips(error.suggestions)}
        </StatusMessage>
      );
    case "week_not_found": {
      // Usually a bye: offer the nearest week this player does have.
      const later = error.availableWeeks.filter((w) => week === null || w > week);
      const suggestion = later[0] ?? error.availableWeeks[error.availableWeeks.length - 1];
      return (
        <StatusMessage
          title={week ? `No projection for week ${week}` : "No projection for this week"}
          body="This is likely a bye week for this player."
        >
          {suggestion !== undefined && (
            <button type="button" className="button" onClick={() => onWeek(suggestion)}>
              Show week {suggestion}
            </button>
          )}
        </StatusMessage>
      );
    }
    default:
      return <RequestError error={error} onRetry={onRetry} />;
  }
}
