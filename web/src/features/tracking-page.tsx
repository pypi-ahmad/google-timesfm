"use client";
import { useState } from "react";
import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, CheckCheck, Plus, Radio, RefreshCw } from "lucide-react";
import { api, json } from "@/lib/api";
import type { Job, Preview, RecordItem, Run } from "@/lib/types";
import { defaultSpec } from "@/lib/spec";
import { dateLabel, shortId } from "@/lib/utils";
import { useAnalyticalContext } from "@/hooks/use-context";
import { useDatasets, useJobs, useRecords, useRun } from "@/hooks/use-records";
import { JobList } from "@/components/jobs";
import { RunViewer } from "@/components/run-viewer";
import { Button } from "@/components/ui/button";
import {
  Badge,
  Check,
  Empty,
  ErrorNotice,
  Field,
  Input,
  Loading,
  PageHeading,
  Section,
  Select,
} from "@/components/ui/controls";

type Tracking = RecordItem<{
  run_id: string;
  auto_refresh: boolean;
  associations: Record<string, string>;
  [key: string]: unknown;
}>;
export function TrackingPage() {
  const context = useAnalyticalContext();
  const client = useQueryClient();
  const runs = useRecords<Run>("runs");
  const tracks = useRecords<Tracking>("tracking");
  const library = useDatasets();
  const jobs = useJobs();
  const [runId, setRunId] = useState(context.runId ?? "");
  const [name, setName] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshPreference, setRefreshPreference] = useState<{
    id: string;
    enabled: boolean;
  } | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [versions, setVersions] = useState<string[]>(context.versions);
  const [associations, setAssociations] = useState<Record<string, string>>({});
  const [notice, setNotice] = useState("");
  const [confirmStop, setConfirmStop] = useState(false);
  const selected = tracks.data?.find((item) => item.id === selectedId);
  const selectedRun = useRun(selected?.payload.run_id ?? null);
  const manifest = selectedRun.data?.payload.manifest as
    { datasets?: { dataset_id: string }[] } | undefined;
  const priorIds = [
    ...new Set((manifest?.datasets ?? []).map((item) => item.dataset_id)),
  ];
  const sourceSpec = selectedRun.data?.payload.spec ?? {};
  const matchingSpec = {
    ...defaultSpec(),
    ...Object.fromEntries(
      Object.keys(defaultSpec())
        .filter((key) => key in sourceSpec)
        .map((key) => [key, sourceSpec[key]]),
    ),
    dataset_version_ids: versions,
  };
  const matching = useQuery({
    queryKey: ["tracking-inputs", selectedId, matchingSpec],
    queryFn: ({ signal }) =>
      api<Preview>(
        `/preview?workspace_id=${encodeURIComponent(context.workspace)}`,
        { method: "POST", body: json(matchingSpec), signal },
      ),
    enabled: !!selectedRun.data && versions.length > 0,
  });
  const currentSeries = (matching.data?.series ?? []).map((item) =>
    String(item.dataset),
  );
  const effectiveAssociations = Object.fromEntries(
    priorIds.flatMap((id) => {
      const chosen = associations[id] ?? (currentSeries.includes(id) ? id : "");
      return currentSeries.includes(chosen) ? [[id, chosen]] : [];
    }),
  );
  const canAssess =
    Object.keys(effectiveAssociations).length > 0 &&
    new Set(Object.values(effectiveAssociations)).size ===
      Object.keys(effectiveAssociations).length &&
    !matching.isFetching;
  const create = useMutation({
    mutationFn: () =>
      api<Tracking>("/tracking", {
        method: "POST",
        body: json({
          workspace_id: context.workspace,
          name: name || `Track ${shortId(runId)}`,
          payload: {
            run_id: runId,
            auto_refresh: autoRefresh,
            associations: {},
          },
        }),
      }),
    onSuccess: (result) => {
      void client.invalidateQueries({ queryKey: ["tracking"] });
      setSelectedId(result.id);
      setNotice(
        "Forecast added to tracking. Choose updated actuals to evaluate it.",
      );
    },
  });
  const action = useMutation({
    mutationFn: (kind: "assess" | "refresh") =>
      api<Job>(`/tracking/${selectedId}/${kind}`, {
        method: "POST",
        body: json({
          dataset_version_ids: versions,
          associations: effectiveAssociations,
        }),
      }),
    onSuccess: (result) => {
      void client.invalidateQueries({ queryKey: ["jobs"] });
      setNotice(
        `${result.kind === "assessment" ? "Actuals assessment" : "Forecast refresh"} queued. The issued forecast remains unchanged.`,
      );
    },
  });
  const stop = useMutation({
    mutationFn: () => api(`/tracking/${selectedId}`, { method: "DELETE" }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["tracking"] });
      setSelectedId("");
      setConfirmStop(false);
      setNotice(
        "Tracking stopped. The saved forecast remains available, subject to your retention policy.",
      );
    },
  });
  const preference = useMutation({
    mutationFn: ({
      tracked,
      enabled,
    }: {
      tracked: Tracking;
      enabled: boolean;
    }) =>
      api<Tracking>(`/tracking/${tracked.id}`, {
        method: "PATCH",
        body: json({
          payload: { ...tracked.payload, auto_refresh: enabled },
          revision: tracked.revision,
        }),
      }),
    onSuccess: (result) => {
      client.setQueryData<Tracking[]>(
        ["tracking", context.workspace],
        (records) =>
          records?.map((record) => (record.id === result.id ? result : record)),
      );
      void client.invalidateQueries({ queryKey: ["tracking"] });
      setNotice("Automatic refresh preference saved.");
    },
    onError: () => {
      void client.invalidateQueries({ queryKey: ["tracking"] });
    },
    onSettled: () => setRefreshPreference(null),
  });
  return (
    <>
      <PageHeading
        title="Tracking"
        description="Evaluate issued forecasts against new actuals and create linked refreshes."
      />
      <div className="grid items-start gap-6 xl:grid-cols-[380px_minmax(0,1fr)]">
        <div className="space-y-5">
          <Section title="Track a saved forecast">
            <form
              onSubmit={(event) => {
                event.preventDefault();
                if (runId) create.mutate();
              }}
              className="space-y-4 p-4"
            >
              <Field label="Issued forecast">
                <Select
                  value={runId}
                  onChange={(event) => setRunId(event.target.value)}
                >
                  <option value="">Select a saved forecast</option>
                  {(runs.data ?? [])
                    .filter((run) => run.payload.kind === "forecast")
                    .map((run) => (
                      <option key={run.id} value={run.id}>
                        {run.name || shortId(run.id)} ·{" "}
                        {dateLabel(run.created_at)}
                      </option>
                    ))}
                </Select>
              </Field>
              <Field label="Tracking name">
                <Input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="e.g. Weekly demand tracking"
                />
              </Field>
              <ErrorNotice error={create.error || runs.error} />
              <Check
                label="Automatically refresh forecasts"
                hint="Off by default. New source versions are assessed automatically; enable this to also issue updated forecasts."
                checked={autoRefresh}
                onChange={(event) => setAutoRefresh(event.target.checked)}
              />
              <Button
                type="submit"
                className="w-full"
                disabled={!runId || create.isPending}
              >
                <Plus />
                Track forecast
              </Button>
            </form>
          </Section>
          <Section title="Tracked forecasts">
            <ErrorNotice error={tracks.error} />
            {tracks.isPending ? (
              <Loading />
            ) : !tracks.data?.length ? (
              <Empty
                icon={<Radio size={21} />}
                title="Follow a forecast over time"
                description="Choose a saved forecast, then upload new actuals as another dataset version."
              />
            ) : (
              <div className="divide-y">
                {tracks.data.map((track) => (
                  <button
                    type="button"
                    key={track.id}
                    className={`w-full px-4 py-3 text-start hover:bg-muted/60 ${track.id === selectedId ? "bg-primary/5" : ""}`}
                    onClick={() => {
                      setSelectedId(track.id);
                      setConfirmStop(false);
                      setAssociations({});
                    }}
                  >
                    <p className="text-xs font-medium">{track.name}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">
                      Issued run {shortId(track.payload.run_id)} ·{" "}
                      {dateLabel(track.created_at)}
                    </p>
                  </button>
                ))}
              </div>
            )}
          </Section>
        </div>
        <div className="space-y-5">
          {notice && (
            <div
              role="status"
              className="rounded-md border border-primary/20 bg-primary/5 p-3 text-xs"
            >
              {notice}
            </div>
          )}
          {selected && (
            <Section
              title={selected.name}
              description="Match updated sources to the dataset identities in the issued forecast."
              actions={
                <Button size="sm" variant="outline" asChild>
                  <Link
                    href={context.href("/forecasts", {
                      run: selected.payload.run_id,
                      dataset: null,
                      target: null,
                      variant: null,
                    })}
                  >
                    Issued forecast
                    <ArrowRight />
                  </Link>
                </Button>
              }
            >
              <div className="space-y-4 p-4">
                <div>
                  <p className="mb-2 text-xs font-medium">
                    Updated actuals versions
                  </p>
                  <div className="max-h-48 overflow-auto rounded-md border px-3 py-2">
                    {library.versions.map((version) => (
                      <Check
                        key={version.id}
                        label={version.name || version.payload.filename}
                        hint={`${shortId(version.id)} · ${dateLabel(version.created_at)}`}
                        checked={versions.includes(version.id)}
                        onChange={(event) =>
                          setVersions(
                            event.target.checked
                              ? [
                                  ...versions.filter(
                                    (id) =>
                                      !library.versions.some(
                                        (item) =>
                                          item.id === id &&
                                          item.payload.dataset_id ===
                                            version.payload.dataset_id,
                                      ),
                                  ),
                                  version.id,
                                ]
                              : versions.filter((id) => id !== version.id),
                          )
                        }
                      />
                    ))}
                    {!library.versions.length && (
                      <p className="py-2 text-xs text-muted-foreground">
                        Upload updated actuals on the Data page.
                      </p>
                    )}
                  </div>
                </div>
                {priorIds.length > 0 && (
                  <div className="space-y-3">
                    {priorIds.map((id) => (
                      <Field key={id} label={`Issued dataset: ${id}`}>
                        <Select
                          value={effectiveAssociations[id] ?? ""}
                          onChange={(event) =>
                            setAssociations((previous) => ({
                              ...previous,
                              [id]: event.target.value,
                            }))
                          }
                        >
                          <option value="">Skip this issued dataset</option>
                          {currentSeries.map((identity) => (
                            <option
                              key={identity}
                              value={identity}
                              disabled={Object.entries(
                                effectiveAssociations,
                              ).some(
                                ([prior, current]) =>
                                  prior !== id && current === identity,
                              )}
                            >
                              {identity}
                            </option>
                          ))}
                        </Select>
                      </Field>
                    ))}
                  </div>
                )}
                <ErrorNotice
                  error={
                    action.error ||
                    stop.error ||
                    library.error ||
                    selectedRun.error ||
                    preference.error ||
                    matching.error
                  }
                />
                <div className="flex flex-wrap gap-2">
                  <Button
                    disabled={!canAssess || action.isPending}
                    onClick={() => action.mutate("assess")}
                  >
                    <CheckCheck />
                    Evaluate against actuals
                  </Button>
                  <Button
                    variant="outline"
                    disabled={!canAssess || action.isPending}
                    onClick={() => action.mutate("refresh")}
                  >
                    <RefreshCw />
                    Refresh forecast
                  </Button>
                </div>
                <p className="text-[10px] leading-5 text-muted-foreground">
                  Assessment scores existing predictions. Refresh submits a new
                  forecast using the saved settings and model provenance.
                </p>
                <Check
                  label="Automatically refresh this forecast"
                  hint="New source versions are assessed automatically. Enabling refresh also submits new forecasts from the saved settings."
                  checked={
                    refreshPreference?.id === selected.id
                      ? refreshPreference.enabled
                      : selected.payload.auto_refresh
                  }
                  disabled={preference.isPending}
                  onChange={(event) => {
                    setRefreshPreference({
                      id: selected.id,
                      enabled: event.target.checked,
                    });
                    preference.mutate({
                      tracked: selected,
                      enabled: event.target.checked,
                    });
                  }}
                />
                <Badge
                  tone={selected.payload.auto_refresh ? "blue" : "neutral"}
                >
                  {selected.payload.auto_refresh
                    ? "Automatic refresh enabled"
                    : "Manual refresh only"}
                </Badge>
                <div className="border-t pt-3">
                  {confirmStop ? (
                    <div className="space-y-3">
                      <p className="text-[11px] leading-5 text-muted-foreground">
                        Stopping removes this forecast’s tracking protection.
                        Your retention policy may subsequently remove eligible
                        results.
                      </p>
                      <div className="flex gap-2">
                        <Button
                          variant="destructive"
                          size="sm"
                          disabled={stop.isPending}
                          onClick={() => stop.mutate()}
                        >
                          Confirm stop tracking
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setConfirmStop(false)}
                        >
                          Keep tracking
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setConfirmStop(true)}
                    >
                      Stop tracking
                    </Button>
                  )}
                </div>
              </div>
            </Section>
          )}
          <Section title="Assessment and refresh jobs">
            <ErrorNotice error={jobs.error} />
            <JobList
              jobs={(jobs.data ?? []).filter(
                (job) =>
                  job.kind === "assessment" ||
                  (job.spec.parent_run_id && job.kind === "forecast"),
              )}
              limit={8}
            />
          </Section>
          {context.runId && <RunViewer />}
        </div>
      </div>
    </>
  );
}
