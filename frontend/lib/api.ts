// Every call to the projections API lives here, so a response-shape change only touches this file.

const BASE_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export const POSITIONS = ["QB", "RB", "WR", "TE"] as const;
export type Position = (typeof POSITIONS)[number];

export function isPosition(value: string | null | undefined): value is Position {
  return POSITIONS.includes(value as Position);
}

export interface Player {
  player_id: string;
  player_name: string;
  position: Position;
  team: string;
}

// Stats that don't apply to a position come back as null (QBs also get null carries).
export interface Stats {
  passing_attempts: number | null;
  passing_yards: number | null;
  passing_tds: number | null;
  interceptions: number | null;
  rushing_yards: number | null;
  rushing_tds: number | null;
  carries: number | null;
  receptions: number | null;
  receiving_yards: number | null;
  receiving_tds: number | null;
}

export interface Projection {
  player: Player;
  week: number;
  opponent: string;
  projected_ppr: number;
  stats: Stats;
  computed_at: string;
}

export interface PlayerProjection extends Projection {
  resolved: { query: string; matched_by: "player_id" | "exact" | "nickname" | "last_name" };
  // Weeks this player has projections for; a missing week is usually a bye.
  available_weeks: number[];
}

export interface RankedProjection extends Projection {
  rank: number;
}

export interface Rankings {
  position: Position;
  week: number;
  limit: number;
  count: number;
  rankings: RankedProjection[];
}

export interface Weeks {
  weeks: number[];
  default_week: number | null;
}

// ---------------------------------------------------------------------------
// Errors: the API answers {"error": {code, message, details}}
// ---------------------------------------------------------------------------

export type ErrorKind =
  | "ambiguous" // 409, details.candidates
  | "player_not_found" // 404, details.suggestions (name lookups only)
  | "week_not_found" // 404, a bye week or a week without projections
  | "unavailable" // 503, or the server could not be reached
  | "other";

export class ApiError extends Error {
  readonly status: number; // 0 for network failures
  readonly code: string;
  readonly kind: ErrorKind;
  readonly candidates: Player[];
  readonly suggestions: Player[];
  readonly availableWeeks: number[];

  constructor(status: number, code: string, message: string, details: Record<string, unknown> = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.kind = errorKind(status, code);
    this.candidates = asArray<Player>(details.candidates);
    this.suggestions = asArray<Player>(details.suggestions);
    this.availableWeeks = asArray<number>(details.available_weeks);
  }
}

function errorKind(status: number, code: string): ErrorKind {
  if (status === 0 || status === 503) return "unavailable";
  if (status === 409) return "ambiguous";
  if (code === "player_not_found") return "player_not_found";
  if (code === "week_not_found" || code === "no_projections") return "week_not_found";
  return "other";
}

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

export function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

type Params = Record<string, string | number | undefined | null>;

async function get<T>(path: string, params: Params = {}, signal?: AbortSignal): Promise<T> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") query.set(key, String(value));
  }
  const qs = query.toString();
  const url = `${BASE_URL}${path}${qs ? `?${qs}` : ""}`;

  let response: Response;
  try {
    response = await fetch(url, { signal, headers: { Accept: "application/json" } });
  } catch (error) {
    if (isAbort(error)) throw error;
    throw new ApiError(0, "network_error", "Could not reach the server.");
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const err = body?.error ?? {};
    throw new ApiError(
      response.status,
      typeof err.code === "string" ? err.code : "http_error",
      typeof err.message === "string" ? err.message : `Request failed (${response.status}).`,
      err.details && typeof err.details === "object" ? err.details : {},
    );
  }
  return body as T;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export function getWeeks(signal?: AbortSignal): Promise<Weeks> {
  return get<Weeks>("/weeks", {}, signal);
}

export async function searchPlayers(q: string, signal?: AbortSignal): Promise<Player[]> {
  const body = await get<{ count: number; players: Player[] }>("/players", { q }, signal);
  return body.players;
}

// Prefer player_id; a name can answer 409 with candidates. Omitting week uses the API's default.
export function getPlayerProjection(
  who: { playerId: string } | { name: string },
  week?: number,
  signal?: AbortSignal,
): Promise<PlayerProjection> {
  const params = "playerId" in who ? { player_id: who.playerId } : { name: who.name };
  return get<PlayerProjection>("/projections/player", { ...params, week }, signal);
}

export function getRankings(
  position: Position,
  week?: number,
  limit = 25,
  signal?: AbortSignal,
): Promise<Rankings> {
  return get<Rankings>("/projections/rankings", { position, week, limit }, signal);
}
