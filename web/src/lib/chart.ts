import type { Row } from "./types";

// Confidence-interval math for forecast-chart.tsx: turns forecast rows into
// contiguous [x, lower, upper] segments that echarts renders as shaded
// bands, without bridging over gaps or invalid intervals.
export type BandPoint = [index: number, lower: number, upper: number];
export function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
// A row's quantiles (q0.1..q0.9) should be non-decreasing; "crossed"
// quantiles indicate an invalid/degenerate interval for that row, which
// bandSegments treats as a break rather than trying to repair it.
export function crossedQuantiles(row: Row): boolean {
  const values = Array.from({ length: 9 }, (_, i) =>
    finiteNumber(row[`q0.${i + 1}`]),
  ).filter((value): value is number => value !== null);
  return values.some((value, i) => i > 0 && value < values[i - 1]);
}
export function bandSegments(
  rows: Row[],
  lower: string,
  upper: string,
  offset = 0,
): BandPoint[][] {
  // `offset` shifts x-indices so a forecast-only band aligns with the
  // combined history+forecast x-axis used by the chart. A new segment
  // starts whenever bounds are missing/non-finite, quantiles are crossed,
  // or `step` isn't contiguous with the previous row — each break is a
  // real gap in the data, not a rendering choice, so segments are never
  // interpolated across them.
  const segments: BandPoint[][] = [];
  let current: BandPoint[] = [];
  let previous: Row | undefined;
  for (const [index, row] of rows.entries()) {
    const lo = finiteNumber(row[lower]);
    const hi = finiteNumber(row[upper]);
    const stepGap =
      previous &&
      typeof row.step === "number" &&
      typeof previous.step === "number" &&
      row.step !== previous.step + 1;
    if (lo === null || hi === null || crossedQuantiles(row) || stepGap) {
      if (current.length) segments.push(current);
      current = [];
    }
    if (lo !== null && hi !== null && !crossedQuantiles(row))
      current.push([index + offset, lo, hi]);
    previous = row;
  }
  if (current.length) segments.push(current);
  return segments;
}
export function axisLabel(row: Row, index: number) {
  return String(row.timestamp ?? row.step ?? index);
}
