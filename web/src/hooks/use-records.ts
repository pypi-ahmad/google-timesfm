"use client";
import { useQueries, useQuery } from "@tanstack/react-query";
import { api, query } from "@/lib/api";
import type { DatasetVersion, Job, RecordItem, Run } from "@/lib/types";
import { useAnalyticalContext } from "./use-context";

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
