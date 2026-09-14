"use client";
import dynamic from "next/dynamic";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, query } from "@/lib/api";
import type { TableData } from "@/lib/types";
import { Button } from "./ui/button";
import {
  Empty,
  ErrorNotice,
  Field,
  Input,
  Loading,
  Section,
} from "./ui/controls";
import { DataTable } from "./data-table";

// Collapsible section (opened on demand from run-viewer.tsx) showing
// nominal-vs-observed interval coverage for a saved run's calibration
// table. The chart is loaded client-only (ssr:false) since echarts needs
// a real DOM to render into.
const CalibrationChart = dynamic(
  () => import("./calibration-chart").then((module) => module.CalibrationChart),
  { ssr: false, loading: () => <Loading label="Loading calibration chart…" /> },
);
function CalibrationContent({
  runId,
  filters,
  horizon,
}: {
  runId: string;
  filters: Record<string, string>;
  horizon: number;
}) {
  // `step` is a 1-based horizon step index (which forecast step to inspect
  // calibration for), clamped to [1, horizon] by the input's onChange.
  const [step, setStep] = useState(1);
  const result = useQuery({
    queryKey: ["calibration", runId, filters, step],
    queryFn: ({ signal }) =>
      api<TableData>(
        `/runs/${encodeURIComponent(runId)}/tables/calibration?${query({ ...filters, step, limit: 20, sort_by: "nominal_coverage_percent" })}`,
        { signal },
      ),
  });
  const rows = result.data?.rows ?? [];
  const hasObserved = rows.some(
    (row) =>
      typeof row.observed_coverage_percent === "number" &&
      Number.isFinite(row.observed_coverage_percent),
  );
  return (
    <>
      <div className="flex flex-wrap items-end justify-between gap-3 border-t px-4 py-3">
        <Field label="Horizon step" className="max-w-32">
          <Input
            type="number"
            min={1}
            max={horizon}
            value={step}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (Number.isInteger(value) && value >= 1 && value <= horizon)
                setStep(value);
            }}
          />
        </Field>
        <p className="text-[10px] text-muted-foreground">
          {[filters.dataset, filters.target, filters.variant]
            .filter(Boolean)
            .join(" · ")}
        </p>
      </div>
      <ErrorNotice error={result.error} />
      {result.isPending ? (
        <Loading />
      ) : hasObserved ? (
        <>
          <CalibrationChart rows={rows} />
          <p className="px-4 pb-3 text-[10px] leading-5 text-muted-foreground">
            Below the diagonal means observed coverage is lower than nominal.
            Counts can be small; these are saved evaluation results.
          </p>
        </>
      ) : (
        <Empty
          title="No scored actuals at this horizon"
          description="Evaluate a holdout or assess a tracked forecast against actuals to inspect observed interval coverage."
        />
      )}
      {rows.length > 0 && (
        <DataTable
          data={rows}
          columns={[
            "nominal_coverage_percent",
            "observed_coverage_percent",
            "observations",
            "mean_width",
            "missing_actuals",
            "invalid_bounds",
            "crossings",
          ].filter((column) => result.data?.columns.includes(column))}
          caption="Saved interval calibration values"
        />
      )}
    </>
  );
}
export function CalibrationPanel(props: {
  runId: string;
  filters: Record<string, string>;
  horizon: number;
}) {
  const [open, setOpen] = useState(false);
  return (
    <Section
      title="Interval calibration"
      description="Compare nominal intervals with observed coverage from saved evaluations."
      actions={
        <Button
          variant="outline"
          size="sm"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        >
          {open ? "Hide coverage" : "Inspect coverage"}
        </Button>
      }
    >
      {open && <CalibrationContent {...props} />}
    </Section>
  );
}
