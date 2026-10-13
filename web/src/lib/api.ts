"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const isForm = typeof FormData !== "undefined" && init?.body instanceof FormData;
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: isForm ? init?.headers : { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function post<T>(path: string, body: unknown = {}): Promise<T> {
  return api<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export function patch<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, { method: "PATCH", body: JSON.stringify(body) });
}

/** Fetch `path`; refetch every `pollMs` while `shouldPoll(data)` is true. */
export function useApi<T>(
  path: string | null,
  options: { pollMs?: number; shouldPoll?: (data: T | null) => boolean } = {},
): { data: T | null; error: string | null; reload: () => void; setData: (d: T) => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const pollRef = useRef(options.shouldPoll);
  useEffect(() => {
    pollRef.current = options.shouldPoll;
  });
  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      try {
        const result = await api<T>(path);
        if (cancelled) return;
        setData(result);
        setError(null);
        if (options.pollMs && pollRef.current?.(result)) timer = setTimeout(load, options.pollMs);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    };
    void load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [path, tick, options.pollMs]);

  return { data, error, reload, setData };
}
