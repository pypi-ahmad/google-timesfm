"use client";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";

// URL-search-params-backed selection state (workspace, dataset versions,
// draft/run/job ids, chart target/dataset/variant, experiment kind) shared
// across every page — the single source of truth for cross-page
// navigation, so switching pages doesn't lose the current selection.
// `update` mutates the current URL; `href` builds a link to another path
// while carrying over (or overriding) these params. Consumed widely; see
// components/shell.tsx for the context bar and hooks/use-draft.ts for how
// draft/version selection ties into the autosave form.
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
