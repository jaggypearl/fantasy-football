"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import RankingsTable from "@/components/RankingsTable";
import StatusMessage, { RequestError } from "@/components/StatusMessage";
import WeekSelect from "@/components/WeekSelect";
import { ApiError, POSITIONS, getRankings, getWeeks, isPosition, type Position } from "@/lib/api";
import { parseWeek } from "@/lib/params";
import { useRequest } from "@/lib/useRequest";

const LIMIT = 25;

export default function RankingsView() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const positionParam = params.get("position")?.toUpperCase();
  const position: Position = isPosition(positionParam) ? positionParam : "QB";
  const urlWeek = parseWeek(params.get("week"));

  const weeks = useRequest("weeks", getWeeks);
  const rankings = useRequest(`${position}:${urlWeek}`, (signal) =>
    getRankings(position, urlWeek ?? undefined, LIMIT, signal),
  );

  const weeksData = weeks.state.status === "success" ? weeks.state.data : null;
  const data = rankings.state.status === "success" ? rankings.state.data : null;
  const week = urlWeek ?? data?.week ?? weeksData?.default_week ?? null;

  function navigate(next: { position: Position; week: number | null }) {
    const query = new URLSearchParams({ position: next.position });
    if (next.week !== null) query.set("week", String(next.week));
    router.replace(`${pathname}?${query}`, { scroll: false });
  }

  function retry() {
    if (weeks.state.status === "error") weeks.retry();
    rankings.retry();
  }

  const error = rankings.state.status === "error" ? rankings.state.error : null;

  return (
    <>
      <header className="page-header">
        <h1 className="page-title">Rankings</h1>
        <p className="page-subtitle">Top {LIMIT} by projected PPR.</p>
      </header>

      <div className="rankings-controls">
        <div className="pill-group" role="group" aria-label="Position">
          {POSITIONS.map((p) => (
            <button
              key={p}
              type="button"
              className="pill"
              aria-pressed={p === position}
              onClick={() => navigate({ position: p, week })}
            >
              {p}
            </button>
          ))}
        </div>
        <WeekSelect
          weeks={weeksData?.weeks ?? null}
          value={week}
          onChange={(w) => navigate({ position, week: w })}
        />
      </div>

      <section aria-live="polite">
        {error instanceof ApiError && error.kind === "week_not_found" ? (
          <StatusMessage title={`No projections for week ${urlWeek ?? ""}`.trim()}>
            <button type="button" className="button" onClick={() => navigate({ position, week: null })}>
              Show the current week
            </button>
          </StatusMessage>
        ) : error ? (
          <RequestError error={error} onRetry={retry} />
        ) : (
          <RankingsTable rows={data?.rankings ?? null} skeletonRows={LIMIT} />
        )}
      </section>
    </>
  );
}
