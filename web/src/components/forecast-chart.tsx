"use client";
import { useEffect, useRef } from "react";
import { useTheme } from "next-themes";
import * as echarts from "echarts/core";
import { LineChart, CustomChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  AriaComponent,
  MarkLineComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption, CustomSeriesOption } from "echarts";
import {
  axisLabel,
  bandSegments,
  crossedQuantiles,
  finiteNumber,
} from "@/lib/chart";
import type { Row } from "@/lib/types";

// Primary forecast visualization: history + forecast line, actuals
// overlay, and shaded 20/40/60/80% confidence bands. Only caller is
// run-viewer.tsx (loaded there via next/dynamic, ssr:false). Band geometry
// comes from lib/chart.ts; this file is the echarts wiring around it.
echarts.use([
  LineChart,
  CustomChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  AriaComponent,
  MarkLineComponent,
  CanvasRenderer,
]);

export function ForecastChart({
  history,
  forecast,
  target,
}: {
  history: Row[];
  forecast: Row[];
  target: string;
}) {
  const element = useRef<HTMLDivElement>(null);
  const { resolvedTheme } = useTheme();
  const invalid = forecast.filter(crossedQuantiles).length;
  useEffect(() => {
    if (!element.current) return;
    const chart = echarts.init(element.current, undefined, {
      renderer: "canvas",
    });
    const dark = resolvedTheme === "dark";
    const foreground = dark ? "#9aa3b4" : "#727b8b";
    const primary = dark ? "#91a4ff" : "#465fd5";
    // x-axis index: `axisLabel` (lib/chart.ts) prefers each row's own
    // timestamp/step, falling back to array position — history and
    // forecast rows share one combined category axis.
    const axis = [...history, ...forecast].map(axisLabel);
    // echarts has no built-in band series that can skip over gaps, so
    // each interval width is drawn as a "custom" series of hand-built
    // polygons — one polygon per contiguous segment from bandSegments,
    // which already excludes crossed/missing-quantile rows (lib/chart.ts).
    // `renderItem` is only invoked with the flattened `data` as a trigger;
    // it draws every segment on its first call and no-ops afterward
    // (`dataIndex !== 0`) since the segments are static per render.
    const bands: CustomSeriesOption[] = [
      ["80% interval", "q0.1", "q0.9", 0.08],
      ["60% interval", "q0.2", "q0.8", 0.13],
      ["40% interval", "q0.3", "q0.7", 0.2],
      ["20% interval", "q0.4", "q0.6", 0.3],
    ].map(([name, lower, upper, opacity]) => {
      const segments = bandSegments(
        forecast,
        String(lower),
        String(upper),
        history.length,
      );
      const flat = segments.flat();
      return {
        name: String(name),
        type: "custom",
        coordinateSystem: "cartesian2d",
        silent: true,
        clip: true,
        animation: false,
        tooltip: { show: false },
        data: flat,
        encode: { x: 0, y: [1, 2] },
        renderItem: (params, api) => {
          if (params.dataIndex !== 0) return;
          return {
            type: "group",
            children: segments.map((segment) => ({
              type: "polygon",
              shape: {
                // A single-point segment has no area to trace as a
                // low-edge/high-edge polygon, so it's drawn as a thin
                // 8px-wide vertical sliver instead, to stay visible.
                points:
                  segment.length === 1
                    ? (() => {
                        const [x, low, high] = segment[0];
                        const lower = api.coord([x, low]);
                        const upper = api.coord([x, high]);
                        return [
                          [lower[0] - 4, lower[1]],
                          [lower[0] + 4, lower[1]],
                          [upper[0] + 4, upper[1]],
                          [upper[0] - 4, upper[1]],
                        ];
                      })()
                    : [
                        ...segment.map(([x, lo]) => api.coord([x, lo])),
                        ...[...segment]
                          .reverse()
                          .map(([x, , hi]) => api.coord([x, hi])),
                      ],
              },
              style: { fill: primary, opacity: Number(opacity) },
            })),
          };
        },
      };
    });
    const historical = history.map((row) =>
      finiteNumber(row.value ?? row.actual ?? row.point),
    );
    const points = [
      ...history.map(() => null),
      ...forecast.map((row) => finiteNumber(row.point)),
    ];
    const actuals = [
      ...history.map(() => null),
      ...forecast.map((row) => finiteNumber(row.actual)),
    ];
    const option: EChartsOption = {
      animation: false,
      backgroundColor: "transparent",
      textStyle: { fontFamily: "Geist, sans-serif", color: foreground },
      aria: {
        enabled: true,
        label: {
          description: `${target || "Selected series"}: ${history.length} historical observations and ${forecast.length} forecast steps. Shaded intervals represent nominal 20, 40, 60, and 80 percent central quantiles. Crossed or missing intervals are omitted, never repaired.`,
        },
      },
      grid: { top: 48, right: 28, bottom: 82, left: 72 },
      tooltip: {
        trigger: "axis",
        confine: true,
        backgroundColor: dark ? "#232835" : "#fff",
        borderColor: dark ? "#3a4252" : "#e6e9ee",
        padding: 12,
        textStyle: {
          color: dark ? "#e5e8f0" : "#20252e",
          fontSize: 13,
          lineHeight: 19,
        },
        valueFormatter: (value) =>
          typeof value === "number"
            ? value.toLocaleString(undefined, { maximumFractionDigits: 4 })
            : "—",
      },
      legend: {
        top: 0,
        left: 20,
        itemWidth: 18,
        itemHeight: 8,
        itemGap: 18,
        textStyle: { color: foreground, fontSize: 12 },
        data: ["Observed", "Forecast", "Actual"],
      },
      xAxis: {
        type: "category",
        data: axis,
        boundaryGap: false,
        axisLine: { lineStyle: { color: dark ? "#3a4252" : "#e6e9ee" } },
        axisTick: { show: false },
        axisLabel: {
          color: foreground,
          fontSize: 12,
          hideOverlap: true,
          // Ticks that look like an ISO date (YYYY-MM-DD...) are shortened
          // to MM-DD to fit; anything else (e.g. a plain row/step number)
          // is shown as-is.
          formatter: (value) =>
            value.length >= 10 && /^\d{4}-\d{2}-\d{2}/.test(value)
              ? value.slice(5, 10)
              : value,
        },
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLabel: { color: foreground, fontSize: 12 },
        splitLine: {
          lineStyle: { color: dark ? "#2b3240" : "#edf0f4", type: "dashed" },
        },
      },
      dataZoom: [
        { type: "inside", filterMode: "none" },
        {
          type: "slider",
          bottom: 8,
          height: 21,
          borderColor: "transparent",
          fillerColor: dark ? "#91a4ff20" : "#465fd512",
          dataBackground: {
            lineStyle: { color: foreground },
            areaStyle: { color: foreground, opacity: 0.07 },
          },
          textStyle: { color: foreground, fontSize: 12 },
        },
      ],
      series: [
        ...bands,
        {
          name: "Observed",
          type: "line",
          data: [...historical, ...forecast.map(() => null)],
          showSymbol: false,
          connectNulls: false,
          lineStyle: { color: dark ? "#c6cedd" : "#8591a5", width: 1.8 },
          itemStyle: { color: "#8591a5" },
        },
        {
          name: "Forecast",
          type: "line",
          data: points,
          showSymbol: forecast.length === 1,
          symbolSize: 6,
          connectNulls: false,
          lineStyle: { color: primary, width: 2 },
          itemStyle: { color: primary },
          // Dashed vertical line at the history/forecast boundary; omitted
          // entirely when there's no history to separate from.
          markLine: history.length
            ? {
                silent: true,
                symbol: "none",
                label: { show: false },
                lineStyle: { color: foreground, type: "dashed", opacity: 0.5 },
                data: [{ xAxis: history.length }],
              }
            : undefined,
        },
        {
          name: "Actual",
          type: "line",
          data: actuals,
          showSymbol: forecast.length === 1,
          connectNulls: false,
          lineStyle: { color: "#23a07a", width: 1.8, type: "dashed" },
          itemStyle: { color: "#23a07a" },
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
  }, [history, forecast, target, resolvedTheme]);
  return (
    <figure>
      <div
        ref={element}
        className="h-[440px] w-full"
        role="img"
        aria-label={`Forecast chart for ${target || "selected series"}`}
      />
      <figcaption className="text-micro flex flex-wrap items-center justify-between gap-3 px-5 pb-5 text-muted-foreground">
        <span className="flex items-center gap-2">
          <span className="flex items-center">
            {[0.1, 0.18, 0.28, 0.4].map((opacity) => (
              <span
                key={opacity}
                className="h-2.5 w-3 bg-primary"
                style={{ opacity }}
              />
            ))}
          </span>
          Nominal 80 / 60 / 40 / 20% intervals
        </span>
        <span>
          {invalid ? `${invalid} crossed intervals omitted · ` : ""}Scroll to
          zoom · drag the range below
        </span>
      </figcaption>
    </figure>
  );
}
