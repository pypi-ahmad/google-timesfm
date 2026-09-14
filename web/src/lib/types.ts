import type { components } from "./generated/api";

// Client-side type aliases layered on the generated OpenAPI types
// (lib/generated/api.d.ts, not hand-edited). `RecordResponse.payload` is
// server-side JSONB with no schema enforced by the generated types, so the
// `payload` shapes below (DatasetVersion, Run, etc.) are conventions this
// UI relies on, not guarantees — treat their fields as best-effort/optional
// when reading real API responses. See lib/api.ts for the fetch wrapper
// and lib/spec.ts for the runtime-validated request contract (Spec).
export type Row = Record<string, unknown>;
export type RecordItem<P = Row> = Omit<
  components["schemas"]["RecordResponse"],
  "payload"
> & { payload: P };
export type DatasetVersion = RecordItem<{
  dataset_id: string;
  filename: string;
  source_name: string;
  columns: string[];
  rows: number;
  artifact: { key: string; sha256: string; size: number };
}>;
export type TableData = components["schemas"]["TablePage"];
export type Job = components["schemas"]["JobResponse"];
export type JobKind = Job["kind"];
export type ApiRunSpec = components["schemas"]["RunSpec"];
export type JobSubmission = components["schemas"]["JobSubmission"];
export type Run = RecordItem<{
  kind: JobKind;
  manifest: Row;
  spec: Row;
  tables: Record<string, { key: string }>;
  export?: unknown;
}>;
export type Preview = {
  quality: Row[];
  series: Row[];
  columns: string[];
  scenario_template: Row[];
  imputation?: Row[];
};
export type ChartData = {
  history: Row[];
  forecast: Row[];
  datasets?: string[];
  targets?: string[];
  variants?: string[];
  dataset?: string;
  target?: string;
  history_sampled?: boolean;
  origin_policy?: string;
};
// Job statuses considered "in flight" for polling/badge purposes; see
// hooks/use-records.ts useJobs (2.5s poll) and components/jobs.tsx.
export const activeStatuses = new Set([
  "pending",
  "queued",
  "running",
  "retrying",
  "cancelling",
  "cancel_requested",
]);
