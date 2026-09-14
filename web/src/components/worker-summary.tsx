"use client";
import { useQuery } from "@tanstack/react-query";
import { Cpu } from "lucide-react";
import { api } from "@/lib/api";
import type { RecordItem, Row } from "@/lib/types";
import { Badge, ErrorNotice, Loading, Section } from "./ui/controls";

// Lists native worker devices registered with the API (used on the Models
// and Overview pages). `worker.payload` is server-defined loose JSON, so
// fields are read defensively rather than assumed present.
export function WorkerSummary() {
  const workers = useQuery({
    queryKey: ["workers"],
    queryFn: ({ signal }) => api<RecordItem<Row>[]>("/workers", { signal }),
    refetchInterval: 15_000,
  });
  // Treats missing `available`/`status` as available — a worker is only
  // excluded when it explicitly reports unavailable or offline.
  const available = (workers.data ?? []).filter(
    (worker) =>
      worker.payload.available !== false && worker.payload.status !== "offline",
  );
  return (
    <Section
      title="Available worker devices"
      description="Checkpoint checks use the CPU worker. Forecast inference uses the configured device."
    >
      <ErrorNotice error={workers.error} />
      {workers.isPending ? (
        <Loading />
      ) : (
        <div className="flex flex-wrap gap-3 p-4">
          {available.length ? (
            available.map((worker) => (
              <div
                key={worker.id}
                className="flex items-center gap-3 rounded-md border px-3 py-2.5"
              >
                <Cpu size={17} className="text-primary" />
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium uppercase">
                      {String(worker.payload.device ?? "worker")}
                    </span>
                    <Badge tone="good">
                      {String(worker.payload.status ?? "available")}
                    </Badge>
                  </div>
                  {typeof worker.payload.vram_free_gb === "number" && (
                    <p className="mt-1 text-[10px] text-muted-foreground">
                      {worker.payload.vram_free_gb.toFixed(1)} GB VRAM free
                      {typeof worker.payload.vram_total_gb === "number"
                        ? ` / ${worker.payload.vram_total_gb.toFixed(1)} GB`
                        : ""}
                    </p>
                  )}
                </div>
              </div>
            ))
          ) : (
            <>
              <Badge tone="warn">No active worker</Badge>
              <p className="text-[11px] text-muted-foreground">
                Start a native worker to process queued model checks and
                forecasts.
              </p>
            </>
          )}
        </div>
      )}
    </Section>
  );
}
