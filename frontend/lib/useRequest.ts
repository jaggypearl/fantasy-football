"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { isAbort } from "./api";

export type RequestState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; error: unknown };

// Runs `fetcher` whenever `key` changes (null means don't fetch), cancelling the previous request.
export function useRequest<T>(key: string | null, fetcher: (signal: AbortSignal) => Promise<T>) {
  const [state, setState] = useState<RequestState<T>>({ status: key ? "loading" : "idle" });
  const [attempt, setAttempt] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    if (key === null) {
      setState({ status: "idle" });
      return;
    }
    const controller = new AbortController();
    setState({ status: "loading" });
    fetcherRef.current(controller.signal).then(
      (data) => setState({ status: "success", data }),
      (error) => {
        if (!isAbort(error)) setState({ status: "error", error });
      },
    );
    return () => controller.abort();
  }, [key, attempt]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  return { state, retry };
}
