import Link from "next/link";
import type { RankedProjection } from "@/lib/api";

interface Props {
  rows: RankedProjection[] | null; // null renders a skeleton
  skeletonRows?: number;
}

export default function RankingsTable({ rows, skeletonRows = 25 }: Props) {
  return (
    <div className="table-wrap">
      <table className="table" aria-busy={rows === null}>
        <thead>
          <tr>
            <th scope="col" className="col-rank">
              #
            </th>
            <th scope="col">Player</th>
            <th scope="col" className="col-team">
              Team
            </th>
            <th scope="col">Opp</th>
            <th scope="col" className="num">
              PPR
            </th>
          </tr>
        </thead>
        <tbody>
          {rows === null
            ? Array.from({ length: skeletonRows }, (_, i) => (
                <tr key={i}>
                  <td className="col-rank tabular">{i + 1}</td>
                  <td>
                    <span className="skeleton skeleton-label" />
                  </td>
                  <td className="col-team">
                    <span className="skeleton skeleton-value" />
                  </td>
                  <td>
                    <span className="skeleton skeleton-value" />
                  </td>
                  <td className="num">
                    <span className="skeleton skeleton-value" />
                  </td>
                </tr>
              ))
            : rows.map((row) => (
                <tr key={row.player.player_id}>
                  <td className="col-rank tabular">{row.rank}</td>
                  <td>
                    <Link
                      className="table-player"
                      href={`/?player_id=${encodeURIComponent(row.player.player_id)}&week=${row.week}`}
                    >
                      {row.player.player_name}
                    </Link>
                    <span className="table-player-team">{row.player.team}</span>
                  </td>
                  <td className="col-team">{row.player.team}</td>
                  <td>{row.opponent}</td>
                  <td className="num col-ppr tabular">{row.projected_ppr.toFixed(1)}</td>
                </tr>
              ))}
        </tbody>
      </table>
    </div>
  );
}
