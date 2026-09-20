"use client";

import { useEffect, useRef } from "react";
import { useTheme } from "next-themes";
import * as echarts from "echarts/core";
import { BarChart, LineChart, ScatterChart } from "echarts/charts";
import {
  AriaComponent,
  DataZoomComponent,
  GridComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts";
import type { PlotData } from "./dataset-explorer";

echarts.use([
  BarChart,
  LineChart,
  ScatterChart,
  AriaComponent,
  DataZoomComponent,
  GridComponent,
  TooltipComponent,
  CanvasRenderer,
]);

export function DatasetChart({
  data,
  kind,
  column,
  x,
  startIndex = 0,
}: {
  data: PlotData;
  kind: string;
  column: string;
  x: string;
  startIndex?: number;
}) {
  const element = useRef<HTMLDivElement>(null);
  const { resolvedTheme } = useTheme();
  useEffect(() => {
    if (!element.current) return;
    const chart = echarts.init(element.current);
    const distribution = kind === "Histogram" || kind === "Category counts";
    const scatter = kind === "Scatter";
    const labels =
      kind === "Histogram"
        ? data.histogram.map(
            (bin) =>
              `${bin.lower.toPrecision(4)} – ${bin.upper.toPrecision(4)}`,
          )
        : kind === "Category counts"
          ? data.categories.map((item) => item.value)
          : data.points.map((point) => String(point.x ?? ""));
    const values =
      kind === "Histogram"
        ? data.histogram.map((bin) => bin.count)
        : kind === "Category counts"
          ? data.categories.map((item) => item.count)
          : data.points.map((point) => point.y);
    const foreground = resolvedTheme === "dark" ? "#9aa3b4" : "#727b8b";
    const option: EChartsOption = {
      animation: false,
      aria: {
        enabled: true,
        label: {
          description: `${kind} chart of ${column}. Values are available in View chart data below.`,
        },
      },
      color: [resolvedTheme === "dark" ? "#91a4ff" : "#465fd5"],
      textStyle: { color: foreground },
      grid: { left: 65, right: 25, top: 40, bottom: 95 },
      tooltip: { trigger: scatter ? "item" : "axis", renderMode: "richText" },
      xAxis: {
        type: scatter ? "value" : "category",
        data: scatter ? undefined : labels,
        name: distribution ? column : x || "Row",
        nameLocation: "middle",
        nameGap: 32,
        axisLabel: { color: foreground, hideOverlap: true },
      },
      yAxis: {
        type: "value",
        name: distribution ? "Count" : column,
        scale: !distribution,
        axisLabel: { color: foreground },
      },
      dataZoom: [
        { type: "inside", startValue: startIndex },
        { type: "slider", bottom: 8, startValue: startIndex },
      ],
      series: [
        {
          type: distribution ? "bar" : scatter ? "scatter" : "line",
          data: scatter
            ? data.points
                .filter(
                  (p) =>
                    typeof p.x === "number" &&
                    Number.isFinite(p.x) &&
                    p.y !== null,
                )
                .map((p) => [p.x as number, p.y])
            : values,
          showSymbol: data.points.length <= 200,
          connectNulls: false,
        },
      ],
    };
    chart.setOption(option);
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [data, kind, column, x, resolvedTheme, startIndex]);
  return (
    <div
      ref={element}
      className="h-96 w-full min-w-0 overflow-hidden"
      role="img"
      aria-label={`${kind} chart of ${column}. Values are available in View chart data below.`}
    />
  );
}
