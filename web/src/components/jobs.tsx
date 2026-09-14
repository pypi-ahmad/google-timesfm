"use client";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import {
  Check,
  CircleStop,
  ExternalLink,
  LoaderCircle,
  RotateCcw,
} from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { dateLabel, shortId } from "@/lib/utils";
import { activeStatuses, type Job } from "@/lib/types";
import { useAnalyticalContext } from "@/hooks/use-context";
import { Button } from "./ui/button";
import { Badge, Empty, ErrorNotice } from "./ui/controls";

// Shared job status/action/list UI (JobStatus badge, JobActions
// cancel/retry/open, JobList) used across the Overview, Models, Tracking,
// and Forecasts pages. Job data comes from hooks/use-records.ts useJobs
// (polled) or a per-job query; activeStatuses is defined in lib/types.ts.
export function JobStatus({ job }: { job: Job }) {
  const active = activeStatuses.has(job.status);
  return (
    <Badge
      tone={
        active
          ? "blue"
          : ["succeeded", "completed"].includes(job.status)
            ? "good"
            : job.status === "failed"
              ? "bad"
              : "neutral"
      }
    >
      {active && <LoaderCircle className="size-3 animate-spin" />}
      {["succeeded", "completed"].includes(job.status) && <Check size={10} />}
      {job.status.replaceAll("_", " ")}
    </Badge>
  );
}
export function JobActions({ job }: { job: Job }) {
  const client = useQueryClient();
  const context = useAnalyticalContext();
  const action = useMutation({
    mutationFn: (name: "cancel" | "retry") =>
      api<Job>(`/jobs/${job.id}/${name}`, { method: "POST" }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["jobs"] });
      void client.invalidateQueries({ queryKey: ["job", job.id] });
    },
  });
  return (
    <div>
      <div className="flex items-center gap-1">
        {activeStatuses.has(job.status) &&
          !["cancelling", "cancel_requested"].includes(job.status) && (
            <Button
              size="sm"
              variant="ghost"
              disabled={action.isPending}
              onClick={() => action.mutate("cancel")}
            >
              <CircleStop />
              Cancel
            </Button>
          )}
        {["failed", "cancelled", "canceled"].includes(job.status) && (
          <Button
            size="sm"
            variant="outline"
            disabled={action.isPending}
            onClick={() => action.mutate("retry")}
          >
            <RotateCcw />
            Retry
          </Button>
        )}
        {job.result_id && (
          <Button size="sm" variant="ghost" asChild>
            {/* Any other job kind (backtest, covariates, etc.) opens the
                shared /experiments route, which picks its specific editor
                from the `experiment` context param. */}
            <Link
              href={context.href(
                job.kind === "scenario"
                  ? "/scenarios"
                  : job.kind === "forecast"
                    ? "/forecasts"
                    : "/experiments",
                {
                  run: job.result_id,
                  job: job.id,
                  dataset: null,
                  target: null,
                  variant: null,
                },
              )}
            >
              <ExternalLink />
              Open
            </Link>
          </Button>
        )}
      </div>
      <ErrorNotice error={action.error} />
    </div>
  );
}
export function JobList({ jobs, limit }: { jobs: Job[]; limit?: number }) {
  const visible = limit ? jobs.slice(0, limit) : jobs;
  if (!visible.length)
    return (
      <Empty
        title="No jobs yet"
        description="Run a forecast or experiment to follow its progress here. Jobs keep their status when you close the browser."
      />
    );
  return (
    <div className="divide-y">
      {visible.map((job) => (
        <div
          key={job.id}
          className="flex flex-wrap items-center justify-between gap-3 px-4 py-3"
        >
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium capitalize">
                {job.kind.replaceAll("_", " ")}
              </span>
              <JobStatus job={job} />
            </div>
            <p className="mt-1.5 text-[10px] text-muted-foreground">
              <span className="font-mono">{shortId(job.id)}</span> ·{" "}
              {job.stage || "Waiting"} · Attempt {job.attempt} ·{" "}
              {dateLabel(job.created_at)}
            </p>
            {job.error && (
              <p className="mt-1 max-w-xl text-[11px] text-destructive">
                {errorMessage(job.error)}
              </p>
            )}
          </div>
          <JobActions job={job} />
        </div>
      ))}
    </div>
  );
}
