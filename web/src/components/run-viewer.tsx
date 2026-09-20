"use client";
import dynamic from "next/dynamic";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Download, Info, LineChart, LockKeyhole } from "lucide-react";
import { api, query } from "@/lib/api";
import { dateLabel, display, shortId } from "@/lib/utils";
import type { ChartData, Row, Run, InputContext } from "@/lib/types";
import { useAnalyticalContext } from "@/hooks/use-context";
import { useRecords, useRun } from "@/hooks/use-records";
import { Button } from "./ui/button";
import {
  Badge,
  Empty,
  ErrorNotice,
  Field,
  Loading,
  Section,
  Select,
} from "./ui/controls";
import { RemoteTable } from "./data-table";
import { CalibrationPanel } from "./calibration-panel";
import {
  RunSummary,
  SignalTracks,
  ReplayCall,
  RunSources,
} from "./run-inspection";
import { Drawer } from "./ui/dialog";

// Displays a saved run: chart + calibration panel + paginated result
// tables (RunViewer), and a separate read-only details panel (RunMetadata)
// used in the run-details drawer/sidebar on features/forecasts-page.tsx.
// Both read from hooks/use-records.ts useRun; RunViewer is also embedded
// directly by features/tracking-page.tsx to show the tracked run.
const ForecastChart = dynamic(
  () => import("./forecast-chart").then((module) => module.ForecastChart),
  { ssr: false, loading: () => <Loading label="Loading chart…" /> },
);
export function RunMetadata({ run }: { run?: Run }) {
  if (!run)
    return (
      <div className="p-4 text-xs leading-6 text-muted-foreground">
        Select a completed run to inspect the model, input versions, and saved
        configuration.
      </div>
    );
  const manifest = run.payload.manifest ?? {};
  const spec = run.payload.spec ?? {};
  // `spec`/`manifest` are server-saved JSON with no enforced schema on the
  // client (run.payload is typed as loose Row fields in lib/types.ts);
  // settings are grouped below by matching key names against regexes
  // rather than a known settings schema, since a run's settings can come
  // from either the spec or an older manifest shape.
  const settings = (spec.settings ?? manifest.settings ?? {}) as Row;
  const formatValue = (value: unknown) =>
    typeof value === "boolean" ? (value ? "Yes" : "No") : display(value);
  const settingGroups = [
    [
      "Forecast",
      [
        ["run type", run.payload.kind] as [string, unknown],
        ...Object.entries(settings).filter(([key]) =>
          /horizon|context|frequency|quantile|interval|mode|task/i.test(key),
        ),
      ],
    ],
    [
      "Model",
      Object.entries(settings).filter(([key]) =>
        /model|checkpoint|revision|device|compile/i.test(key),
      ),
    ],
    [
      "Inputs",
      Object.entries(settings).filter(([key]) =>
        /dataset|target|covariate|group|column|version/i.test(key),
      ),
    ],
  ] as const;
  const groupedKeys = new Set(
    settingGroups.flatMap(([, entries]) => entries.map(([key]) => key)),
  );
  const remaining = Object.entries(settings).filter(
    ([key]) => !groupedKeys.has(key),
  );
  return (
    <div className="space-y-5 p-4">
      <div className="rounded-lg border p-3 text-sm">
        <p>Recorded runtime: {display(manifest.runtime_seconds)} seconds</p>
        <p>
          Timing:{" "}
          {manifest.runtime_scope === "predict_batch"
            ? "Model prediction call"
            : manifest.runtime_scope === "analysis_execution"
              ? "Complete analysis execution"
              : "Scope not recorded"}
        </p>
        <p>Device: {display(manifest.device)}</p>
        <p>Checkpoint: {display(manifest.checkpoint)}</p>
        <p>Code revision: {display(manifest.repository_revision)}</p>
        <p>
          Disabled signals:{" "}
          {((manifest.disabled_covariates ?? []) as string[]).join(", ") ||
            "None recorded"}
        </p>
      </div>
      <RunSources run={run} />
      <details>
        <summary className="cursor-pointer font-medium">
          Input hashes, roles, shapes and warnings
        </summary>
        <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-all rounded bg-muted p-3 text-xs">
          {JSON.stringify(
            {
              source_file_hashes: manifest.datasets,
              roles: manifest.mapping ?? spec.mapping,
              windows: manifest.input_windows ?? "Not recorded",
              preparation: manifest.preparation,
              warnings: manifest.warnings ?? [],
              chunking: manifest.chunking,
            },
            null,
            2,
          )}
        </pre>
        <p className="text-caption text-muted-foreground">
          Hashes identify source files, not model tensors. Shapes describe
          prepared arrays before model interpolation.
        </p>
      </details>
      <ReplayCall key={run.id} run={run} />
      <div>
        <div className="mb-2 flex items-center gap-2 text-xs font-medium">
          <LockKeyhole size={13} />
          Saved run
        </div>
        <p className="text-micro break-all font-mono">{run.id}</p>
        <p className="text-micro mt-1 text-muted-foreground">
          {dateLabel(run.created_at)}
        </p>
      </div>
      {[...settingGroups, ["Other settings", remaining] as const].map(
        ([title, entries]) =>
          entries.length > 0 && (
            <section key={title} className="space-y-3">
              <h3 className="text-micro font-semibold uppercase tracking-wide text-muted-foreground">
                {title}
              </h3>
              {entries.map(([key, value]) => (
                <div
                  key={key}
                  className="text-caption flex items-start justify-between gap-3"
                >
                  <span className="capitalize text-muted-foreground">
                    {key.replaceAll("_", " ")}
                  </span>
                  <span className="max-w-32 break-words text-end tabular">
                    {formatValue(value)}
                  </span>
                </div>
              ))}
            </section>
          ),
      )}
      <details className="text-caption">
        <summary className="cursor-pointer font-medium">
          Model provenance
        </summary>
        <pre className="text-micro mt-2 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-muted p-3 font-mono">
          {JSON.stringify(
            manifest.model_provenance ?? spec.model ?? {},
            null,
            2,
          )}
        </pre>
      </details>
      <details className="text-caption">
        <summary className="cursor-pointer font-medium">
          Full manifest and specification
        </summary>
        <pre className="text-micro mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-muted p-3 font-mono">
          {JSON.stringify({ manifest, spec }, null, 2)}
        </pre>
      </details>
      {/* Native browser download, deliberately bypassing the api() fetch
          wrapper in lib/api.ts — this needs the browser to handle the
          response as a file save, not a parsed JSON body. */}
      <Button className="w-full" variant="outline" asChild>
        <a href={`/api/v1/runs/${run.id}/export`} download>
          <Download />
          Export run bundle
        </a>
      </Button>
      <p className="text-micro text-muted-foreground">
        Submitted settings and source versions are immutable. Edit a draft to
        create another run.
      </p>
    </div>
  );
}
export function RunViewer() {
  const context = useAnalyticalContext();
  const runs = useRecords<Run>("runs");
  const selected = useRun(context.runId);
  const [tableName, setTableName] = useState("");
  const [comparison, setComparison] = useState({ key: "", reference: "" });
  const selectionKey = `${context.runId}:${context.dataset}:${context.target}:${context.variant}`;
  const [zoom, setZoom] = useState({ key: "", view: "full" });
  const chartView = zoom.key === selectionKey ? zoom.view : "full";
  const reference = comparison.key === selectionKey ? comparison.reference : "";
  const run = selected.data;
  const hasChart = ["forecast", "predictions", "comparisons"].some(
    (name) => name in (run?.payload.tables ?? {}),
  );
  const filters = {
    dataset: context.dataset,
    target: context.target,
    variant: context.variant,
  };
  const chart = useQuery({
    queryKey: ["run-chart", context.runId, filters, reference],
    queryFn: ({ signal }) =>
      api<ChartData>(
        `/runs/${context.runId}/chart?${query({ ...filters, reference })}`,
        {
          signal,
        },
      ),
    enabled: !!run && hasChart,
  });
  const tables = Object.keys(run?.payload.tables ?? {});
  const activeTable = tables.includes(tableName)
    ? tableName
    : (tables.find((name) => name === "forecast") ?? tables[0]);
  const spec = run?.payload.spec as
    { mapping?: { targets?: string[] } } | undefined;
  // The chart endpoint's targets/datasets/variants arrays are optional; if
  // absent, they're derived by scanning the returned rows for distinct
  // values (falling back further to the run's saved target mapping for
  // targets specifically). Union with the spec's targets covers series
  // that have no forecast rows yet.
  const targets = [
    ...new Set([
      ...(chart.data?.targets ?? []),
      ...(spec?.mapping?.targets ?? []),
      ...(chart.data?.forecast ?? [])
        .map((row) => String(row.target ?? ""))
        .filter(Boolean),
    ]),
  ];
  const datasets = chart.data?.datasets ?? [
    ...new Set(
      [...(chart.data?.forecast ?? []), ...(chart.data?.history ?? [])]
        .map((row) => String(row.dataset ?? ""))
        .filter(Boolean),
    ),
  ];
  const variants = chart.data?.variants ?? [
    ...new Set(
      (chart.data?.forecast ?? [])
        .map((row) => String(row.variant ?? ""))
        .filter(Boolean),
    ),
  ];
  const shownTarget = context.target || targets[0] || "";
  const shownDataset = context.dataset || datasets[0] || "";
  const shownVariant = context.variant || variants[0] || "";
  const inputContext = useQuery({
    queryKey: ["input-context", run?.id, shownDataset, shownVariant],
    queryFn: ({ signal }) =>
      api<InputContext>(
        `/runs/${run!.id}/input-context?${query({ dataset: shownDataset, variant: shownVariant })}`,
        { signal },
      ),
    enabled: !!run && !!chart.data,
  });
  // A row missing a target/dataset/variant field entirely is treated as
  // "not applicable to this filter" and passes through regardless of the
  // selected value — only rows that carry the field and disagree with the
  // current selection are filtered out.
  const chartRows = (rows: Row[]) =>
    rows.filter(
      (row) =>
        (!shownTarget || !row.target || row.target === shownTarget) &&
        (!shownDataset || !row.dataset || row.dataset === shownDataset) &&
        (!shownVariant || !row.variant || row.variant === shownVariant),
    );
  return (
    <div className="space-y-4">
      <Section
        title="Forecast result"
        actions={
          run && (
            <div className="flex gap-2">
              <Badge tone="good">Saved result</Badge>
              <Drawer
                title="Execution report"
                trigger={
                  <Button size="sm" variant="outline">
                    Execution report
                  </Button>
                }
              >
                <RunMetadata run={run} />
              </Drawer>
            </div>
          )
        }
      >
        <div className="border-b p-4">
          <Field label="Saved run">
            <Select
              aria-label="Saved run"
              value={context.runId ?? ""}
              onChange={(event) =>
                context.update({
                  run: event.target.value,
                  target: null,
                  dataset: null,
                  variant: null,
                })
              }
            >
              <option value="">Select a completed run</option>
              {(runs.data ?? []).map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name || `${item.payload.kind} · ${shortId(item.id)}`} ·{" "}
                  {dateLabel(item.created_at)}
                </option>
              ))}
            </Select>
          </Field>
        </div>
        <ErrorNotice error={selected.error || runs.error} />
        {!run ? (
          <Empty
            icon={<LineChart size={22} />}
            title="Your next forecast starts here"
            description="Choose a dataset version, assign column roles, and preview data quality. Completed results stay available across sessions."
          />
        ) : (
          <>
            <div className="grid grid-cols-1 gap-3 px-4 py-3 sm:grid-cols-3">
              <Field label="Target">
                <Select
                  value={shownTarget}
                  aria-label="Chart target"
                  onChange={(event) =>
                    context.update({ target: event.target.value })
                  }
                >
                  {targets.map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                  {!targets.length && <option value="">All targets</option>}
                </Select>
              </Field>
              <Field label="Dataset">
                <Select
                  aria-label="Chart dataset"
                  value={shownDataset}
                  onChange={(event) =>
                    context.update({
                      dataset: event.target.value,
                      target: null,
                    })
                  }
                >
                  {!datasets.length && (
                    <option value="">First available</option>
                  )}
                  {datasets.map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                </Select>
              </Field>
              <Field label="Variant">
                <Select
                  aria-label="Chart variant"
                  value={shownVariant}
                  onChange={(event) =>
                    context.update({ variant: event.target.value })
                  }
                >
                  {!variants.length && <option value="">Default</option>}
                  {variants.map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                </Select>
              </Field>
            </div>
            <ErrorNotice error={chart.error} />
            {run && (
              <RunSummary
                run={run}
                filters={{
                  dataset: shownDataset,
                  target: shownTarget,
                  variant: shownVariant,
                  reference,
                }}
              />
            )}
            {!!chart.data?.references?.length && (
              <div className="px-4 pb-3">
                <Field label="Reference curve">
                  <Select
                    value={reference}
                    onChange={(event) =>
                      setComparison({
                        key: selectionKey,
                        reference: event.target.value,
                      })
                    }
                  >
                    <option value="">Hide reference</option>
                    {chart.data.references.map((name) => (
                      <option key={name} value={name}>
                        {name.replaceAll("_", " ")}
                      </option>
                    ))}
                  </Select>
                </Field>
              </div>
            )}
            {chart.data?.origin != null && (
              <p className="px-4 text-caption text-muted-foreground">
                Displayed forecast origin: {chart.data.origin}
              </p>
            )}
            {!hasChart ? (
              <Empty
                title="This run contains tabular output"
                description="Inspect its result tables and saved metadata below."
              />
            ) : chart.isPending ? (
              <Loading label="Loading series…" />
            ) : chart.data && chart.data.forecast.length ? (
              <ForecastChart
                history={chartRows(chart.data.history)}
                forecast={chartRows(chart.data.forecast)}
                target={shownTarget}
                reference={chart.data.reference_rows ?? []}
                referenceName={reference}
                events={inputContext.data?.events ?? []}
                view={chartView}
                onViewChange={(view) => setZoom({ key: selectionKey, view })}
              />
            ) : (
              <Empty
                title="No chart for this selection"
                description="Inspect the result tables below, or choose another dataset and target."
              />
            )}
            {/* history_sampled/origin_policy are server-set flags describing
                how the chart endpoint reduced the data for display only —
                they don't affect what's in the result tables below. */}
            {(chart.data?.history_sampled ||
              chart.data?.origin_policy === "latest") && (
              <p className="px-4 pb-3 text-[10px] text-muted-foreground">
                {chart.data.history_sampled
                  ? "Historical display uses reduced resolution. Source observations remain available through Inspect dataset. "
                  : ""}
                {chart.data.origin_policy === "latest"
                  ? "Rolling results show the latest origin; all origins are available in the tables."
                  : ""}
              </p>
            )}
          </>
        )}
      </Section>
      <ErrorNotice error={inputContext.error} />
      {inputContext.data && (
        <SignalTracks
          key={`${run?.id}:${shownDataset}:${shownVariant}`}
          context={inputContext.data}
          timeline={[
            ...(chart.data?.history ?? []),
            ...(chart.data?.forecast ?? []),
          ]}
          startIndex={
            chartView === "horizon" ? (chart.data?.history.length ?? 0) : 0
          }
        />
      )}
      {run && tables.includes("calibration") && (
        <CalibrationPanel
          key={run.id}
          runId={run.id}
          filters={{
            dataset: shownDataset,
            target: shownTarget,
            variant: shownVariant,
          }}
          horizon={Number(
            (run.payload.spec.settings as Row | undefined)?.horizon ?? 1,
          )}
        />
      )}
      {run && activeTable && (
        <Section
          title="Result tables"
          description="Server-paged rows, with sorting across the complete result."
          actions={
            <Select
              className="max-w-40"
              aria-label="Result table"
              value={activeTable}
              onChange={(event) => setTableName(event.target.value)}
            >
              {tables.map((name) => (
                <option key={name} value={name}>
                  {name.replaceAll("_", " ")}
                </option>
              ))}
            </Select>
          }
        >
          <RemoteTable
            key={`${run.id}:${activeTable}:${JSON.stringify(filters)}`}
            runId={run.id}
            tableName={activeTable}
            filters={filters}
          />
        </Section>
      )}
      {run && (
        <div className="flex items-start gap-2 rounded-md bg-primary/5 px-4 py-3 text-[11px] leading-5 text-muted-foreground">
          <Info size={14} className="mt-0.5 shrink-0 text-primary" />
          Intervals describe nominal model quantiles. Inspect calibration and
          observed coverage before using them as confidence guarantees.
        </div>
      )}
    </div>
  );
}
