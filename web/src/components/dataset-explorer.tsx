"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { useQuery } from "@tanstack/react-query";
import { api, query } from "@/lib/api";
import type { DatasetVersion, Row, TableData } from "@/lib/types";
import { DataTable } from "./data-table";
import { Button } from "./ui/button";
import { ErrorNotice, Field, Loading, Select } from "./ui/controls";

const DatasetChart = dynamic(
  () => import("./dataset-chart").then((m) => m.DatasetChart),
  { ssr: false },
);

type Profile = {
  rows: number;
  columns: Row[];
  numeric_columns: string[];
  missing_cells: number;
  duplicate_rows: number;
  memory_bytes: number;
  statistics: Row[];
  correlation_columns: string[];
  correlations: Row[];
};
export type PlotData = {
  numeric: boolean;
  total: number;
  sampled: boolean;
  points: { x: unknown; y: number | null }[];
  histogram: { lower: number; upper: number; count: number }[];
  categories: { value: string; count: number }[];
  non_null: number;
};

export function DatasetExplorer({ version }: { version: DatasetVersion }) {
  const [view, setView] = useState("Rows");
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(10);
  const [column, setColumn] = useState(version.payload.columns[0] ?? "");
  const [x, setX] = useState("");
  const [kind, setKind] = useState("Line");
  const base = `/datasets/versions/${encodeURIComponent(version.id)}`;
  const rows = useQuery({
    queryKey: ["dataset-preview", version.id, offset, limit],
    queryFn: ({ signal }) =>
      api<TableData>(`${base}/preview?${query({ offset, limit })}`, { signal }),
    enabled: view === "Rows",
  });
  const profile = useQuery({
    queryKey: ["dataset-profile", version.id],
    queryFn: ({ signal }) => api<Profile>(`${base}/profile`, { signal }),
    enabled: view !== "Rows",
    staleTime: Infinity,
  });
  const plot = useQuery({
    queryKey: ["dataset-plot", version.id, column, x],
    queryFn: ({ signal }) =>
      api<PlotData>(`${base}/plot?${query({ column, x: x || undefined })}`, {
        signal,
      }),
    enabled: view === "Charts" && !!column,
    staleTime: Infinity,
  });
  const total = rows.data?.total ?? version.payload.rows;
  const summary = profile.data;
  return (
    <div className="space-y-4 p-4">
      <div className="flex flex-wrap gap-2" aria-label="Dataset explorer views">
        {["Rows", "Overview", "Statistics", "Charts", "Correlations"].map(
          (item) => (
            <Button
              key={item}
              size="sm"
              variant={view === item ? "default" : "outline"}
              aria-pressed={view === item}
              onClick={() => setView(item)}
            >
              {item}
            </Button>
          ),
        )}
      </div>
      {view === "Rows" ? (
        <>
          <div className="flex flex-wrap items-end gap-2">
            <Field label="Rows per page">
              <Select
                value={limit}
                onChange={(e) => {
                  setLimit(Number(e.target.value));
                  setOffset(0);
                }}
              >
                {[5, 10, 25, 100, 1000].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </Select>
            </Field>
            <Button variant="outline" size="sm" onClick={() => setOffset(0)}>
              Head
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setOffset(Math.max(0, total - limit))}
            >
              Tail
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - limit))}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={offset + limit >= total}
              onClick={() => setOffset(offset + limit)}
            >
              Next
            </Button>
          </div>
          <ErrorNotice error={rows.error} />
          {rows.isPending ? (
            <Loading />
          ) : (
            rows.data && (
              <>
                <p className="text-caption text-muted-foreground">
                  Rows {total ? offset + 1 : 0}–
                  {Math.min(offset + limit, total)} of {total.toLocaleString()}.
                  Column sorting applies to this page only.
                </p>
                <DataTable
                  key={`${offset}:${limit}`}
                  data={rows.data.rows}
                  columns={rows.data.columns}
                  caption="Dataset rows"
                />
              </>
            )
          )}
        </>
      ) : (
        <>
          <ErrorNotice error={profile.error} />
          {profile.isPending ? (
            <Loading label="Profiling the full dataset…" />
          ) : (
            summary && (
              <>
                <p className="text-caption text-muted-foreground">
                  Summaries use all {summary.rows.toLocaleString()} rows of this
                  source version. Exploration does not change your forecast
                  inputs.
                </p>
                {view === "Overview" && (
                  <>
                    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                      {[
                        ["Columns", summary.columns.length],
                        ["Missing cells", summary.missing_cells],
                        ["Duplicate rows", summary.duplicate_rows],
                        [
                          "Memory (MiB)",
                          (summary.memory_bytes / 1048576).toFixed(2),
                        ],
                      ].map(([label, value]) => (
                        <div key={label} className="rounded-lg border p-3">
                          <p className="text-caption text-muted-foreground">
                            {label}
                          </p>
                          <p className="mt-1 text-lg font-semibold">
                            {value?.toLocaleString()}
                          </p>
                        </div>
                      ))}
                    </div>
                    <DataTable
                      data={summary.columns}
                      caption="Column types and data quality"
                    />
                    <p className="text-caption text-muted-foreground">
                      Duplicate rows count repeats after the first occurrence.
                      Missing values and infinities are reported separately;
                      neither is filled or removed.
                    </p>
                  </>
                )}
                {view === "Statistics" && (
                  <>
                    <p className="text-caption text-muted-foreground">
                      Like df.describe(): count, mean, standard deviation,
                      minimum, quartiles, and maximum for numeric columns.
                      Missing and infinite values are excluded.
                    </p>
                    {summary.statistics.length ? (
                      <DataTable
                        data={summary.statistics}
                        caption="Descriptive statistics"
                      />
                    ) : (
                      <p>
                        No numeric columns to summarize. Use Charts for category
                        counts.
                      </p>
                    )}
                  </>
                )}
                {view === "Correlations" && (
                  <>
                    <p className="text-caption text-muted-foreground">
                      Pearson correlations, using available finite pairs (at
                      least two). Blank values indicate insufficient data or a
                      constant column. Correlation does not imply causation.
                    </p>
                    {summary.numeric_columns.length > 32 && (
                      <p>
                        Showing the first 32 numeric columns of{" "}
                        {summary.numeric_columns.length}.
                      </p>
                    )}
                    {summary.correlations.length ? (
                      <DataTable
                        data={summary.correlations}
                        caption="Pearson correlation matrix"
                      />
                    ) : (
                      <p>No numeric columns to correlate.</p>
                    )}
                  </>
                )}
                {view === "Charts" && (
                  <>
                    <div className="grid gap-3 sm:grid-cols-3">
                      <Field label="Column / Y axis">
                        <Select
                          value={column}
                          onChange={(e) => setColumn(e.target.value)}
                        >
                          {version.payload.columns.map((name) => (
                            <option key={name}>{name}</option>
                          ))}
                        </Select>
                      </Field>
                      <Field label="Chart type">
                        <Select
                          value={kind}
                          onChange={(e) => setKind(e.target.value)}
                        >
                          {[
                            "Line",
                            "Histogram",
                            "Scatter",
                            "Category counts",
                          ].map((name) => (
                            <option key={name}>{name}</option>
                          ))}
                        </Select>
                      </Field>
                      {(kind === "Line" || kind === "Scatter") && (
                        <Field label="X axis">
                          <Select
                            value={x}
                            onChange={(e) => setX(e.target.value)}
                          >
                            <option value="">Row number (zero-based)</option>
                            {(kind === "Scatter"
                              ? summary.numeric_columns
                              : version.payload.columns
                            ).map((name) => (
                              <option key={name}>{name}</option>
                            ))}
                          </Select>
                        </Field>
                      )}
                    </div>
                    <ErrorNotice error={plot.error} />
                    {plot.isPending ? (
                      <Loading />
                    ) : (
                      plot.data && (
                        <>
                          {kind !== "Category counts" && !plot.data.numeric ? (
                            <p>
                              Choose a numeric Y column, or switch to Category
                              counts.
                            </p>
                          ) : kind === "Scatter" &&
                            x &&
                            !summary.numeric_columns.includes(x) ? (
                            <p>
                              Choose a numeric X column for a scatter chart.
                            </p>
                          ) : (
                            <DatasetChart
                              data={plot.data}
                              kind={kind}
                              column={column}
                              x={x}
                            />
                          )}
                          <p className="text-caption text-muted-foreground">
                            {kind === "Line" || kind === "Scatter"
                              ? `${plot.data.sampled ? "Evenly sampled to at most 2,000 rows" : "All rows"}; source row order is preserved. Missing or infinite Y values are gaps; scatter plots omit invalid pairs.`
                              : kind === "Histogram"
                                ? "20 bins counted across all finite values in the full dataset."
                                : "Top 20 non-missing values counted across the full dataset; other values are omitted."}
                          </p>
                          <details>
                            <summary className="cursor-pointer text-sm">
                              View chart data
                            </summary>
                            <DataTable
                              data={
                                kind === "Histogram"
                                  ? plot.data.histogram
                                  : kind === "Category counts"
                                    ? plot.data.categories
                                    : plot.data.points
                              }
                              caption="Chart data"
                            />
                          </details>
                        </>
                      )
                    )}
                  </>
                )}
              </>
            )
          )}
        </>
      )}
    </div>
  );
}
