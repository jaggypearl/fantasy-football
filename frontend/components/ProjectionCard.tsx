import type { PlayerProjection, Stats } from "@/lib/api";

// Display order and labels. Keyed by Stats so a renamed API field fails the build here.
const STAT_LABELS: Record<keyof Stats, string> = {
  passing_attempts: "Pass attempts",
  passing_yards: "Passing yards",
  passing_tds: "Passing TDs",
  interceptions: "Interceptions",
  carries: "Carries",
  rushing_yards: "Rushing yards",
  rushing_tds: "Rushing TDs",
  receptions: "Receptions",
  receiving_yards: "Receiving yards",
  receiving_tds: "Receiving TDs",
};

// TDs and interceptions are small fractions (0.04), so they keep two decimals.
const TWO_DECIMALS = new Set<keyof Stats>(["passing_tds", "interceptions", "rushing_tds", "receiving_tds"]);

function formatStat(key: keyof Stats, value: number): string {
  return value.toFixed(TWO_DECIMALS.has(key) ? 2 : 1);
}

export default function ProjectionCard({ projection }: { projection: PlayerProjection }) {
  const { player, week, opponent, projected_ppr, stats } = projection;
  const rows = (Object.keys(STAT_LABELS) as (keyof Stats)[]).filter((key) => stats[key] !== null);

  return (
    <article className="card" aria-label={`${player.player_name} projection`}>
      <div className="card-header">
        <div>
          <h2 className="card-name">{player.player_name}</h2>
          <p className="card-meta">
            {player.position} · {player.team}
          </p>
          <p className="card-meta">
            Week {week} · vs {opponent}
          </p>
        </div>
        <div className="card-ppr">
          <span className="card-ppr-value tabular">{projected_ppr.toFixed(1)}</span>
          <span className="card-ppr-label">Projected PPR</span>
        </div>
      </div>
      {rows.length > 0 && (
        <dl className="stats">
          {rows.map((key) => (
            <div key={key} className="stat-row">
              <dt>{STAT_LABELS[key]}</dt>
              <dd className="tabular">{formatStat(key, stats[key] as number)}</dd>
            </div>
          ))}
        </dl>
      )}
    </article>
  );
}

export function ProjectionCardSkeleton() {
  return (
    <div className="card" aria-busy="true" aria-label="Loading projection">
      <div className="card-header">
        <div>
          <span className="skeleton skeleton-title" />
          <span className="skeleton skeleton-line" />
          <span className="skeleton skeleton-line" />
        </div>
        <span className="skeleton skeleton-number" />
      </div>
      <div className="stats">
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i} className="stat-row">
            <span className="skeleton skeleton-label" />
            <span className="skeleton skeleton-value" />
          </div>
        ))}
      </div>
    </div>
  );
}
