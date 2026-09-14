"use client";
import { useState } from "react";
import { useFormContext, useWatch } from "react-hook-form";
import type { Preview, Row } from "@/lib/types";
import type { Spec } from "@/lib/spec";
import { Badge, Check, Field, Section, Select } from "./ui/controls";
import { Button } from "./ui/button";
import { DataTable } from "./data-table";

// Read-only display of the last successful /preview response (quality
// report, prepared/interpolated data samples, per-series exclusion
// toggles). Owned by features/forecasts-page.tsx, which runs the preview
// mutation and passes its result down; see lib/types.ts Preview.
export function InputPreview({
  preview,
  // Whether this preview still reflects the form's current values —
  // false once the spec has changed since the preview was taken, which
  // features/forecasts-page.tsx also uses to gate job submission.
  current,
}: {
  preview: Preview;
  current: boolean;
}) {
  const [tab, setTab] = useState<"quality" | "series" | "imputation">(
    "quality",
  );
  const [seriesIndex, setSeriesIndex] = useState(0);
  const { control, setValue } = useFormContext<Spec>();
  const excluded = useWatch({ control, name: "preparation.excluded_groups" });
  const groups = useWatch({ control, name: "preparation.group_columns" });
  const selected =
    preview.series[Math.min(seriesIndex, preview.series.length - 1)];
  const preparedRows = (selected?.preview ?? []) as Row[];
  // A "blocked" row that's already excluded (via the group-exclusion
  // toggle below) no longer counts against submission — only unresolved
  // blocking issues are surfaced.
  const blocked = preview.quality.filter(
    (row) => row.status === "blocked" && !row.excluded,
  ).length;
  return (
    <Section
      title="Data quality preview"
      actions={
        <Badge tone={current ? "good" : "warn"}>
          {blocked
            ? `${blocked} blocked`
            : current
              ? "Current inputs"
              : "Inputs changed"}
        </Badge>
      }
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b px-3 py-2">
        <div className="flex flex-wrap gap-1">
          {(
            [
              { value: "quality", label: "Quality report" },
              { value: "series", label: "Prepared data" },
              { value: "imputation", label: "Interpolation" },
            ] as const
          ).map((item) => (
            <Button
              key={item.value}
              type="button"
              size="sm"
              variant={tab === item.value ? "default" : "ghost"}
              onClick={() => setTab(item.value)}
            >
              {item.label}
            </Button>
          ))}
        </div>
        <span className="text-[10px] text-muted-foreground">
          {preview.series.length} prepared series
        </span>
      </div>
      {tab === "quality" && (
        <DataTable data={preview.quality} caption="Data quality report" />
      )}
      {tab === "imputation" && (
        <>
          <p className="px-4 py-3 text-[11px] leading-5 text-muted-foreground">
            Inspect values interpolated inside the model context. Uploaded
            source data is preserved.
          </p>
          <DataTable
            data={preview.imputation ?? []}
            caption="Model-context interpolation preview"
          />
        </>
      )}
      {tab === "series" && (
        <>
          <div className="border-b p-3">
            <Field label="Prepared series">
              <Select
                value={seriesIndex}
                onChange={(event) => setSeriesIndex(Number(event.target.value))}
              >
                {preview.series.map((row, index) => (
                  <option value={index} key={index}>
                    {String(row.dataset ?? `Series ${index + 1}`)} ·{" "}
                    {String(row.rows)} rows{row.excluded ? " · excluded" : ""}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <DataTable data={preparedRows} caption="Prepared dataset rows" />
          <p className="border-t px-4 py-2 text-[10px] text-muted-foreground">
            First {preparedRows.length} prepared rows. Grouping and generated
            calendar features are included.
          </p>
        </>
      )}
      {groups.length > 0 && preview.series.length > 0 && (
        <details className="border-t p-3">
          <summary className="cursor-pointer text-[11px]">
            Exclude groups from this run
          </summary>
          <div className="mt-2 max-h-40 overflow-auto">
            {preview.series.map((row, index) => {
              const id = String(row.dataset ?? row.dataset_id ?? index);
              return (
                <Check
                  key={id}
                  label={id}
                  checked={excluded.includes(id)}
                  onChange={(event) =>
                    setValue(
                      "preparation.excluded_groups",
                      event.target.checked
                        ? [...excluded, id]
                        : excluded.filter((value) => value !== id),
                      { shouldDirty: true },
                    )
                  }
                />
              );
            })}
          </div>
        </details>
      )}
    </Section>
  );
}
