"use client";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";

export function useAnalyticalContext() {
  const params = useSearchParams();
  const pathname = usePathname();
  const router = useRouter();
  const search = params.toString();
  const update = useCallback(
    (changes: Record<string, string | null>, path = pathname) => {
      const next = new URLSearchParams(search);
      for (const [key, value] of Object.entries(changes)) {
        if (value) next.set(key, value);
        else next.delete(key);
      }
      router.replace(`${path}${next.size ? `?${next}` : ""}`, {
        scroll: false,
      });
    },
    [pathname, router, search],
  );
  const href = useCallback(
    (path: string, changes: Record<string, string | null> = {}) => {
      const next = new URLSearchParams(search);
      for (const [key, value] of Object.entries(changes)) {
        if (value) next.set(key, value);
        else next.delete(key);
      }
      return `${path}${next.size ? `?${next}` : ""}`;
    },
    [search],
  );
  return {
    workspace: params.get("workspace") || "local",
    versions: (params.get("versions") || params.get("version") || "")
      .split(",")
      .filter(Boolean),
    draftId: params.get("draft"),
    runId: params.get("run"),
    jobId: params.get("job"),
    target: params.get("target") || "",
    dataset: params.get("dataset") || "",
    variant: params.get("variant") || "",
    experiment: params.get("experiment") || "backtest",
    update,
    href,
  };
}
