"use client";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  ArrowUpRight,
  CircleHelp,
  Cpu,
  Database,
  FileClock,
  LineChart,
  Plus,
  Radio,
} from "lucide-react";
import { api } from "@/lib/api";
import {
  activeStatuses,
  type Row,
  type Run,
  type TableData,
} from "@/lib/types";
import { dateLabel, shortId } from "@/lib/utils";
import { useAnalyticalContext } from "@/hooks/use-context";
import { useDatasets, useJobs, useRecords } from "@/hooks/use-records";
import { JobList } from "@/components/jobs";
import { DataTable } from "@/components/data-table";
import { Button } from "@/components/ui/button";
import {
  Badge,
  Empty,
  ErrorNotice,
  Loading,
  PageHeading,
  Section,
} from "@/components/ui/controls";

export function OverviewPage() {
  const context = useAnalyticalContext();
  const library = useDatasets();
  const jobs = useJobs();
  const runs = useRecords<Run>("runs");
  const drafts = useRecords("drafts");
  const tracking = useRecords("tracking");
  const latestEvaluation = runs.data?.find((run) => {
    const settings = run.payload.spec.settings as Row | undefined;
    return (
      "metrics" in run.payload.tables &&
      ([
        "assessment",
        "backtest",
        "baselines",
        "settings",
        "joint_independent",
        "covariates",
        "anomaly",
      ].includes(run.payload.kind) ||
        settings?.task === "holdout")
    );
  });
  const accuracy = useQuery({
    queryKey: ["overview-accuracy", latestEvaluation?.id],
    queryFn: ({ signal }) =>
      api<TableData>(`/runs/${latestEvaluation!.id}/tables/metrics?limit=4`, {
        signal,
      }),
    enabled: !!latestEvaluation,
  });
  const scoredRows = (accuracy.data?.rows ?? []).filter(
    (row) => typeof row.mae === "number" && Number.isFinite(row.mae),
  );
  const workers = useQuery({
    queryKey: ["workers"],
    queryFn: ({ signal }) => api<Row[]>("/workers", { signal }),
    refetchInterval: 15_000,
  });
  const active = (jobs.data ?? []).filter((job) =>
    activeStatuses.has(job.status),
  );
  const workerStatus = (worker: Row) =>
    String(
      (worker.payload as Row | undefined)?.status ??
        worker.status ??
        "Registered",
    );
  const isStaleWorker = (worker: Row) =>
    /stale|offline|expired/i.test(workerStatus(worker));
  const currentWorkers = (workers.data ?? []).filter(
    (worker) => !isStaleWorker(worker),
  );
  const staleWorkers = (workers.data ?? []).filter(isStaleWorker);
  const workerRow = (worker: Row, index: number) => (
    <div key={String(worker.id ?? index)} className="px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-medium">
          {String(worker.name ?? worker.id ?? `Worker ${index + 1}`)}
        </p>
        {isStaleWorker(worker) && <Badge tone="warn">Stale</Badge>}
      </div>
      <p className="text-micro mt-1 text-muted-foreground">
        {workerStatus(worker)} ·{" "}
        {String(
          (worker.payload as Row | undefined)?.device ??
            worker.device ??
            "Device reported by worker",
        )}
        {typeof (worker.payload as Row | undefined)?.vram_free_gb ===
          "number" &&
          ` · ${Number((worker.payload as Row).vram_free_gb).toFixed(1)} GB VRAM free`}
      </p>
    </div>
  );
  return (
    <>
      <PageHeading
        title="Overview"
        description="Pick up an investigation, check active jobs, or start with a new dataset."
        actions={
          <Button asChild>
            <Link href={context.href("/forecasts", { draft: null, job: null })}>
              <Plus />
              New forecast
            </Link>
          </Button>
        }
      />
      <div className="mb-6 grid grid-cols-[repeat(auto-fit,minmax(min(100%,220px),1fr))] gap-4">
        {[
          {
            label: "Dataset versions",
            value: library.isLoading ? "—" : library.versions.length,
            note: `${library.datasets.length} datasets in your library`,
            icon: Database,
            href: "/data",
          },
          {
            label: "Saved runs",
            value: runs.isPending ? "—" : (runs.data?.length ?? "—"),
            note: "Immutable results and configurations",
            icon: LineChart,
            href: "/forecasts",
          },
          {
            label: "Active jobs",
            value: jobs.isPending ? "—" : active.length,
            note: active.length
              ? `${active.filter((job) => job.status === "running").length} currently running`
              : "Ready for your next run",
            icon: Cpu,
            href: "/overview",
          },
          {
            label: "Saved drafts",
            value: drafts.isPending ? "—" : (drafts.data?.length ?? "—"),
            note: "Continue where you left off",
            icon: FileClock,
            href: "/forecasts",
          },
        ].map((item) => (
          <Link
            key={item.label}
            href={context.href(item.href)}
            className="panel px-4 py-4 hover:border-primary/30"
          >
            <div className="text-caption flex items-center justify-between text-muted-foreground">
              <span>{item.label}</span>
              <item.icon size={15} />
            </div>
            <p className="metric my-3 text-3xl font-semibold">{item.value}</p>
            <p className="text-micro text-muted-foreground">{item.note}</p>
          </Link>
        ))}
      </div>
      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1.65fr)_minmax(280px,1fr)]">
        <div className="space-y-5">
          <Section
            title="Latest evaluated accuracy"
            description={
              latestEvaluation
                ? `${latestEvaluation.name || latestEvaluation.payload.kind} · ${shortId(latestEvaluation.id)} · ${dateLabel(latestEvaluation.created_at)}`
                : "Saved scores from a holdout, experiment, or actuals assessment."
            }
            actions={
              latestEvaluation && (
                <Button variant="outline" size="sm" asChild>
                  <Link
                    href={context.href("/forecasts", {
                      run: latestEvaluation.id,
                      dataset: null,
                      target: null,
                      variant: null,
                    })}
                  >
                    View evaluated run
                    <ArrowUpRight />
                  </Link>
                </Button>
              )
            }
          >
            <ErrorNotice error={accuracy.error} />
            {runs.isPending || (latestEvaluation && accuracy.isPending) ? (
              <Loading />
            ) : scoredRows.length ? (
              <>
                <DataTable
                  data={scoredRows}
                  columns={[
                    "dataset",
                    "target",
                    "variant",
                    "scope",
                    "step",
                    "origin",
                    "observations",
                    "mae",
                    "rmse",
                    "smape_percent",
                  ].filter((column) => accuracy.data?.columns.includes(column))}
                  caption="Latest saved accuracy metrics"
                />
                <p className="px-4 py-3 text-[10px] leading-5 text-muted-foreground">
                  Previewing {scoredRows.length} of {accuracy.data?.total} saved
                  metric rows. MAE and RMSE use each target’s units; sMAPE is a
                  percentage.
                </p>
              </>
            ) : (
              <Empty
                title="No scored evaluation yet"
                description="Run a holdout or backtest, or assess a tracked forecast against actuals to see saved accuracy here."
              />
            )}
          </Section>
          <Section
            title="Job activity"
            description="Jobs continue independently of this browser."
          >
            <ErrorNotice error={jobs.error} />
            {jobs.isPending ? (
              <Loading />
            ) : (
              <JobList jobs={jobs.data ?? []} limit={4} />
            )}
          </Section>
          <Section
            title="Recent runs"
            actions={
              <Link
                href={context.href("/forecasts")}
                className="flex items-center gap-1 text-[11px] text-primary"
              >
                View all runs
                <ArrowUpRight size={12} />
              </Link>
            }
          >
            <ErrorNotice error={runs.error} />
            {runs.isPending ? (
              <Loading />
            ) : runs.data?.length ? (
              <div className="divide-y">
                {runs.data.slice(0, 4).map((run) => (
                  <Link
                    key={run.id}
                    href={context.href("/forecasts", {
                      run: run.id,
                      target: null,
                      dataset: null,
                      variant: null,
                    })}
                    className="flex items-center justify-between gap-3 px-4 py-3 hover:bg-muted/50"
                  >
                    <div>
                      <p className="text-xs font-medium">
                        {run.name || `${run.payload.kind} · ${shortId(run.id)}`}
                      </p>
                      <p className="mt-1 text-[10px] text-muted-foreground">
                        {dateLabel(run.created_at)} ·{" "}
                        {Object.keys(run.payload.tables ?? {}).length} result
                        tables
                      </p>
                    </div>
                    <Badge tone="good">{run.payload.kind}</Badge>
                  </Link>
                ))}
              </div>
            ) : (
              <Empty
                title="Results will live here"
                description="Every completed forecast and experiment saves its settings, provenance, and exportable result tables."
              />
            )}
          </Section>
        </div>
        <div className="space-y-5">
          <Section title="Active tracking">
            <ErrorNotice error={tracking.error} />
            <Link
              href={context.href("/tracking")}
              className="flex items-center justify-between gap-3 px-4 py-4 hover:bg-muted/50"
            >
              <div className="flex items-center gap-3">
                <Radio size={18} className="text-primary" />
                <div>
                  <p className="text-xs font-medium">
                    {tracking.isPending ? "—" : (tracking.data?.length ?? "—")}{" "}
                    tracked forecasts
                  </p>
                  <p className="mt-1 text-[10px] text-muted-foreground">
                    Assess actuals and manage refresh preferences
                  </p>
                </div>
              </div>
              <ArrowUpRight size={14} className="text-muted-foreground" />
            </Link>
          </Section>
          <Section title="Continue a draft">
            <ErrorNotice error={drafts.error} />
            {drafts.isPending ? (
              <Loading />
            ) : drafts.data?.length ? (
              <div className="divide-y">
                {drafts.data.slice(0, 5).map((draft) => (
                  <Link
                    key={draft.id}
                    href={context.href("/forecasts", { draft: draft.id })}
                    className="flex items-center justify-between gap-3 px-4 py-3 hover:bg-muted/50"
                  >
                    <div className="min-w-0">
                      <p className="truncate text-xs font-medium">
                        {draft.name}
                      </p>
                      <p className="mt-1 text-[10px] text-muted-foreground">
                        Revision {draft.revision} ·{" "}
                        {dateLabel(draft.created_at)}
                      </p>
                    </div>
                    <ArrowRight size={14} className="text-muted-foreground" />
                  </Link>
                ))}
              </div>
            ) : (
              <div className="p-4 text-xs leading-6 text-muted-foreground">
                Drafts save automatically as you configure a forecast. Come back
                here to resume them.
              </div>
            )}
          </Section>
          <Section
            title="Worker availability"
            description="Native worker processes registered with the API."
          >
            <ErrorNotice error={workers.error} />
            {workers.isPending ? (
              <Loading />
            ) : workers.data?.length ? (
              <div>
                <div className="divide-y">{currentWorkers.map(workerRow)}</div>
                {staleWorkers.length > 0 && (
                  <details className="border-t">
                    <summary className="cursor-pointer px-4 py-3 text-sm font-medium">
                      Show {staleWorkers.length} stale{" "}
                      {staleWorkers.length === 1 ? "worker" : "workers"}
                    </summary>
                    <div className="divide-y border-t">
                      {staleWorkers.map(workerRow)}
                    </div>
                  </details>
                )}
              </div>
            ) : (
              <div className="p-4">
                <Badge tone="warn">No worker heartbeat</Badge>
                <p className="mt-3 text-[11px] leading-6 text-muted-foreground">
                  Start the native worker to process queued jobs. Dataset
                  previews and saved results remain available.
                </p>
              </div>
            )}
          </Section>
          <div className="rounded-lg border border-primary/15 bg-primary/5 p-4">
            <CircleHelp size={17} className="mb-3 text-primary" />
            <h2 className="text-xs font-semibold">
              From data to a reliable forecast
            </h2>
            <ol className="mt-3 space-y-2 text-[11px] leading-5 text-muted-foreground">
              <li>1. Upload data and choose column roles.</li>
              <li>2. Preview quality and run a holdout.</li>
              <li>3. Compare experiments, then track actuals.</li>
            </ol>
            <Button className="mt-4" variant="outline" size="sm" asChild>
              <Link href={context.href("/data")}>
                Open data library
                <ArrowRight />
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}
