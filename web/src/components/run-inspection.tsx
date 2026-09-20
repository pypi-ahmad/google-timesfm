"use client";

import { useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import dynamic from "next/dynamic";
import { api, query } from "@/lib/api";
import { display } from "@/lib/utils";
import type { InputContext, InputWindow, Row, Run } from "@/lib/types";
import { useAnalyticalContext } from "@/hooks/use-context";
import { Button } from "./ui/button";
import { Check, ErrorNotice, Loading, Section } from "./ui/controls";
import { DataTable } from "./data-table";
import type { PlotData } from "./dataset-explorer";

const DatasetChart = dynamic(
  () => import("./dataset-chart").then((m) => m.DatasetChart),
  { ssr: false },
);

export function InputTimeline({ windows }: { windows: InputWindow[] }) {
  if (!windows.length)
    return (
      <p className="p-4 text-sm text-muted-foreground">
        Input context is not recorded for this selection. New runs record their
        prepared inputs.
      </p>
    );
  return (
    <div className="space-y-4 p-4">
      {windows.map((w, index) => (
        <div key={index} className="space-y-3">
          <p className="text-sm font-medium">
            {w.dataset}
            {w.variant ? ` · ${w.variant}` : ""} · {w.context_length} historical
            rows → {w.horizon} forecast rows
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr>
                  <th className="p-2">Signal</th>
                  <th className="p-2">History</th>
                  <th className="p-2">Forecast window</th>
                  <th className="p-2">Missing inputs</th>
                </tr>
              </thead>
              <tbody>
                {w.signals.map((s) => (
                  <tr key={s.signal} className="border-t">
                    <th className="p-2 font-medium">{s.signal}</th>
                    <td className="p-2">
                      <span className="rounded bg-primary/10 px-2 py-1">
                        {s.used === false
                          ? "Omitted by inference mode / chunking"
                          : s.role === "target"
                            ? "Target history"
                            : "Observed signal"}
                      </span>
                    </td>
                    <td className="p-2">
                      <span
                        className={`rounded px-2 py-1 ${s.role === "known_future" ? "bg-emerald-500/10" : "bg-muted"}`}
                      >
                        {s.used === false
                          ? "Not used by this variant"
                          : s.role === "known_future"
                            ? "Known future input"
                            : s.role === "target"
                              ? "To predict — not an input"
                              : "Not supplied"}
                      </span>
                    </td>
                    <td className="p-2">{s.missing}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-caption text-muted-foreground">
            Prepared inputs before model interpolation. Shapes: targets{" "}
            {w.context_shape.join(" × ")}; past-only{" "}
            {w.past_only_shape?.join(" × ") ?? "none"}; known-future{" "}
            {w.past_future_shape?.join(" × ") ?? "none"}.
            {w.sampled
              ? " Historical displays are sampled to at most 2,000 rows per signal."
              : ""}
          </p>
          {!!w.lineage.length && (
            <details>
              <summary className="cursor-pointer text-sm">
                Preparation operations
              </summary>
              <pre className="overflow-auto text-xs">
                {JSON.stringify(w.lineage, null, 2)}
              </pre>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}

type Summary = {
  sensitivity?: Row[];
  metrics: Row | null;
  scope?: string;
  period_start?: string;
  period_end?: string;
  delta: {
    total: number;
    percent: number | null;
    largest_increase: number;
    largest_decrease: number;
    matched_steps: number;
    complete: boolean;
    reference: string;
  } | null;
};

export function RunSummary({
  run,
  filters,
}: {
  run: Run;
  filters: Record<string, string>;
}) {
  const result = useQuery({
    queryKey: ["run-summary", run.id, filters],
    queryFn: ({ signal }) =>
      api<Summary>(`/runs/${run.id}/summary?${query(filters)}`, { signal }),
  });
  const summary = result.data;
  return (
    <div className="space-y-3 p-4">
      <ErrorNotice error={result.error} />
      {result.isPending ? (
        <Loading label="Loading saved scores…" />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {[
              ["MAE", "mae"],
              ["RMSE", "rmse"],
              ["sMAPE (%)", "smape_percent"],
              ["Observations", "observations"],
            ].map(([label, field]) => (
              <div key={field} className="rounded-lg border p-3">
                <p className="text-caption text-muted-foreground">{label}</p>
                <p className="mt-1 text-xl font-semibold">
                  {display(summary?.metrics?.[field])}
                </p>
              </div>
            ))}
          </div>
          {!!summary?.sensitivity?.length && (
            <details>
              <summary className="cursor-pointer text-sm">
                Measured signal sensitivity
              </summary>
              <p className="text-caption text-muted-foreground">
                Saved error differences from removal experiments, not causal
                contributions.
              </p>
              <DataTable
                data={summary.sensitivity}
                caption="Signal removal error differences"
              />
            </details>
          )}
          <p className="text-caption text-muted-foreground">
            {summary?.metrics
              ? `${summary.scope === "all_evaluated_windows" ? "All evaluated windows; the chart shows the latest origin" : summary.scope === "observed_actuals" ? "Observed actuals assessment" : "Holdout evaluation"}. ${summary.period_start ?? ""} – ${summary.period_end ?? ""}. Scores use full saved observations, not chart samples.`
              : "Awaiting actuals — no evaluated accuracy is recorded for this selection."}
          </p>
          {summary?.metrics?.excluded !== undefined && (
            <p className="text-caption">
              Excluded forecast pairs: {display(summary.metrics.excluded)}
            </p>
          )}
          {summary?.delta && run.payload.kind === "scenario" && (
            <div className="rounded-lg border bg-primary/5 p-4">
              <p className="font-medium">
                Predicted change vs {summary.delta.reference}
              </p>
              <div className="mt-2 grid grid-cols-2 gap-3">
                <span>Total change: {display(summary.delta.total)}</span>
                <span>Change: {display(summary.delta.percent)}%</span>
                <span>
                  Largest signed step change:{" "}
                  {display(summary.delta.largest_increase)}
                </span>
                <span>
                  Smallest signed step change:{" "}
                  {display(summary.delta.largest_decrease)}
                </span>
              </div>
              <p className="mt-2 text-caption">
                {summary.delta.matched_steps} matched steps ·{" "}
                {summary.delta.complete
                  ? "Complete overlap"
                  : "Incomplete overlap; totals cover matched steps only"}
                . Model sensitivity, not causal impact. Values are in target
                units.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function SignalTracks({
  context,
  timeline,
  startIndex = 0,
}: {
  context: InputContext;
  timeline: Row[];
  startIndex?: number;
}) {
  const [chosen, setChosen] = useState<string[]>([]);
  const available = [
    ...new Set(
      context.rows
        .filter((row) => row.role !== "target")
        .map((row) => String(row.signal)),
    ),
  ];
  return (
    <Section
      title="Input timeline and signals"
      description="Saved prepared inputs for the displayed origin. Select up to three signal tracks."
    >
      <InputTimeline windows={context.windows} />
      {!!available.length && (
        <div className="flex flex-wrap gap-4 border-t px-4">
          {available.map((signal) => (
            <Check
              key={signal}
              label={signal}
              checked={chosen.includes(signal)}
              disabled={!chosen.includes(signal) && chosen.length >= 3}
              onChange={(e) =>
                setChosen(
                  e.target.checked
                    ? [...chosen, signal]
                    : chosen.filter((s) => s !== signal),
                )
              }
            />
          ))}
        </div>
      )}
      {chosen
        .filter((s) => available.includes(s))
        .map((signal) => {
          const rows = context.rows.filter((row) => row.signal === signal);
          const byTime = new Map(
            rows.map((row) => [
              String(row.timestamp ?? row.position),
              row.value,
            ]),
          );
          const data: PlotData = {
            numeric: true,
            total: rows.length,
            sampled: context.windows.some((w) => w.sampled),
            points: timeline.map((row, index) => {
              const x = row.timestamp ?? row.step ?? index;
              const value = byTime.get(String(x));
              return { x, y: typeof value === "number" ? value : null };
            }),
            histogram: [],
            categories: [],
            non_null: rows.filter((row) => row.value != null).length,
          };
          return (
            <div key={signal} className="min-w-0 border-t p-4">
              <p className="font-medium">{signal}</p>
              <DatasetChart
                data={data}
                kind="Line"
                column={signal}
                x="Saved timeline"
                startIndex={startIndex}
              />
              <details>
                <summary className="cursor-pointer text-sm">
                  View signal values
                </summary>
                <DataTable data={rows} caption={`${signal} input values`} />
              </details>
            </div>
          );
        })}
    </Section>
  );
}

export function ReplayCall({ run }: { run: Run }) {
  const [script, setScript] = useState("");
  const [status, setStatus] = useState("");
  return (
    <details
      className="space-y-3"
      onToggle={async (event) => {
        if (event.currentTarget.open && !script) {
          try {
            const response = await fetch(`/api/v1/runs/${run.id}/replay`);
            if (!response.ok)
              throw new Error("Replay is unavailable for this saved run.");
            setScript(await response.text());
          } catch (error) {
            setStatus(String(error));
          }
        }
      }}
    >
      <summary className="cursor-pointer font-medium">Show the call</summary>
      <p className="text-caption text-muted-foreground">
        Running this script creates a new job on this workbench. It requires the
        original dataset versions and checkpoint; loading this panel does not
        run inference.
      </p>
      <pre className="max-h-80 overflow-auto rounded bg-muted p-3 text-xs">
        {script || status || "Loading…"}
      </pre>
      <div className="flex gap-2">
        <Button
          type="button"
          size="sm"
          disabled={!script}
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(script);
              setStatus("Copied");
            } catch {
              setStatus(
                "Clipboard unavailable; select the code or download it.",
              );
            }
          }}
        >
          Copy Python
        </Button>
        <Button size="sm" variant="outline" asChild>
          <a href={`/api/v1/runs/${run.id}/replay`} download>
            Download Python
          </a>
        </Button>
      </div>
      <p role="status" className="text-caption">
        {status}
      </p>
    </details>
  );
}

export function RunSources({ run }: { run: Run }) {
  const context = useAnalyticalContext();
  const ids = (run.payload.spec?.dataset_version_ids ?? []) as string[];
  return (
    <div className="flex flex-wrap gap-2">
      {ids.map((id, index) => (
        <Button key={id} size="sm" variant="outline" asChild>
          <Link
            href={context.href("/data", {
              versions: id,
              workspace: run.workspace_id,
            })}
          >
            Inspect dataset {index + 1}
          </Link>
        </Button>
      ))}
    </div>
  );
}
