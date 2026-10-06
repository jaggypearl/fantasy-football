import type { ReactNode } from "react";
import { ApiError } from "@/lib/api";

interface Props {
  title: string;
  body?: ReactNode;
  tone?: "neutral" | "error";
  children?: ReactNode; // actions
}

export default function StatusMessage({ title, body, tone = "neutral", children }: Props) {
  return (
    <div className="status" data-tone={tone} role={tone === "error" ? "alert" : "status"}>
      <p className="status-title">{title}</p>
      {body && <p className="status-body">{body}</p>}
      {children && <div className="status-actions">{children}</div>}
    </div>
  );
}

// The server-unreachable and unexpected-error cases, shared by both pages.
export function RequestError({ error, onRetry }: { error: unknown; onRetry: () => void }) {
  const unavailable = !(error instanceof ApiError) || error.kind === "unavailable";
  return (
    <StatusMessage
      tone={unavailable ? "neutral" : "error"}
      title={unavailable ? "The server is waking up, try again in a few seconds" : "Something went wrong"}
      body={unavailable ? undefined : (error as ApiError).message}
    >
      <button type="button" className="button" onClick={onRetry}>
        Retry
      </button>
    </StatusMessage>
  );
}
