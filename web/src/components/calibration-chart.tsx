"use client";
import { useEffect, useRef } from "react";
import { useTheme } from "next-themes";
import * as echarts from "echarts/core";
import { LineChart, ScatterChart } from "echarts/charts";
import {
  AriaComponent,
  GridComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { Row } from "@/lib/types";

echarts.use([
  LineChart,
  ScatterChart,
  AriaComponent,
  GridComponent,
  TooltipComponent,
  CanvasRenderer,
]);

export function CalibrationChart({ rows }: { rows: Row[] }) {
  const host = useRef<HTMLDivElement>(null);
  const { resolvedTheme } = useTheme();
  useEffect(() => {
    if (!host.current) return;
    const dark = resolvedTheme === "dark";
    const muted = dark ? "#9aa3b4" : "#727b8b";
    const chart = echarts.init(host.current);
    const points = rows
      .filter(
        (row) =>
          typeof row.observed_coverage_percent === "number" &&
          Number.isFinite(row.observed_coverage_percent),
      )
      .map((row) => [
        row.nominal_coverage_percent,
        row.observed_coverage_percent,
        row.observations,
      ]);
    chart.setOption({
      animation: false,
      aria: {
        enabled: true,
        label: {
          description:
            "Observed coverage against nominal interval coverage, read directly from the saved calibration table. The diagonal marks equal nominal and observed coverage.",
        },
      },
      grid: { top: 18, right: 24, bottom: 52, left: 68 },
      textStyle: { fontFamily: "Geist, sans-serif", color: muted },
      tooltip: { trigger: "item", confine: true },
      xAxis: {
        type: "value",
        name: "Nominal coverage (%)",
        nameLocation: "middle",
        nameGap: 30,
        min: 0,
        max: 100,
        axisLabel: { color: muted, fontSize: 10 },
        splitLine: { lineStyle: { color: dark ? "#2b3240" : "#edf0f4" } },
      },
      yAxis: {
        type: "value",
        name: "Observed coverage (%)",
        nameLocation: "middle",
        nameGap: 44,
        min: 0,
        max: 100,
        axisLabel: { color: muted, fontSize: 10 },
        splitLine: { lineStyle: { color: dark ? "#2b3240" : "#edf0f4" } },
      },
      series: [
        {
          type: "line",
          name: "Nominal reference",
          data: [
            [0, 0],
            [100, 100],
          ],
          showSymbol: false,
          silent: true,
          lineStyle: { color: muted, type: "dashed", width: 1 },
        },
        {
          type: "scatter",
          name: "Saved observed coverage",
          dimensions: [
            "Nominal coverage (%)",
            "Observed coverage (%)",
            "Observations",
          ],
          encode: { x: 0, y: 1, tooltip: [0, 1, 2] },
          data: points,
          symbolSize: 10,
          itemStyle: { color: dark ? "#91a4ff" : "#465fd5" },
        },
      ],
    });
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [rows, resolvedTheme]);
  return (
    <div
      ref={host}
      className="h-64 w-full"
      role="img"
      aria-label="Nominal versus observed interval coverage"
    />
  );
}
