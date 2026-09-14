"use client";
import { useQueries, useQuery } from "@tanstack/react-query";
import { api, query } from "@/lib/api";
import type { DatasetVersion, Job, RecordItem, Run } from "@/lib/types";
import { useAnalyticalContext } from "./use-context";

// react-query wrappers over the workspace-scoped list/detail record
// endpoints (datasets, jobs, runs, drafts, models, tracking, activity).
// All queries key on the current workspace from use-context.ts, so
// switching workspaces naturally invalidates/refetches the right data.
export function useRecords<T = RecordItem>(resource: string) {
  const { workspace } = useAnalyticalContext();
  return useQuery({
    queryKey: [resource, workspace],
    queryFn: ({ signal }) =>
      api<T[]>(`/${resource}?${query({ workspace_id: workspace })}`, {
        signal,
      }),
  });
}
export function useDatasets() {
  const datasets = useRecords("datasets");
  // One request per dataset to fetch its versions (no batched "all
  // versions" endpoint). Fine at this app's expected dataset counts; would
  // need a combined endpoint if that assumption stops holding.
  const versions = useQueries({
    queries: (datasets.data ?? []).map((item) => ({
      queryKey: ["versions", item.id],
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        api<DatasetVersion[]>(
          `/datasets/${encodeURIComponent(item.id)}/versions`,
          { signal },
        ),
    })),
  });
  return {
    datasets: datasets.data ?? [],
    versions: versions.flatMap((item) => item.data ?? []),
    isLoading: datasets.isPending || versions.some((item) => item.isPending),
    error: datasets.error || versions.find((item) => item.error)?.error,
  };
}
export function useJobs() {
  // Polls every 2.5s so job status/progress stays current without a push
  // channel; components/jobs.tsx and activeStatuses (lib/types.ts) decide
  // when a job is still "active" for badges/disabled actions.
  const { workspace } = useAnalyticalContext();
  return useQuery({
    queryKey: ["jobs", workspace],
    queryFn: ({ signal }) =>
      api<Job[]>(`/jobs?${query({ workspace_id: workspace })}`, { signal }),
    refetchInterval: 2500,
  });
}
export function useRun(id: string | null) {
  return useQuery({
    queryKey: ["run", id],
    queryFn: ({ signal }) => api<Run>(`/runs/${id}`, { signal }),
    enabled: !!id,
  });
}
