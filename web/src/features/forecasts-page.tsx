"use client";
import { useEffect, useRef, useState } from "react";
import { FormProvider } from "react-hook-form";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  CheckCheck,
  Copy,
  FileClock,
  Info,
  Play,
  RotateCcw,
  Save,
  SlidersHorizontal,
} from "lucide-react";
import { api, json } from "@/lib/api";
import { defaultSpec, schemaMessage, type Spec } from "@/lib/spec";
import type { Job, JobKind, JobSubmission, Preview } from "@/lib/types";
import { useAnalyticalContext } from "@/hooks/use-context";
import { useDraft } from "@/hooks/use-draft";
import { useRecords, useRun } from "@/hooks/use-records";
import { InputPreview } from "@/components/input-preview";
import { ForecastFields } from "@/components/forecast-form";
import {
  ConfigurationEditor,
  ScenarioEditor,
} from "@/components/experiment-editors";
import { JobActions, JobStatus } from "@/components/jobs";
import { RunMetadata, RunViewer } from "@/components/run-viewer";
import { Button } from "@/components/ui/button";
import {
  Badge,
  ErrorNotice,
  Loading,
  PageHeading,
  Select,
} from "@/components/ui/controls";
import { Drawer } from "@/components/ui/dialog";

// The core forecast/experiment/scenario builder+viewer, rendered by
// app/[page]/page.tsx for /forecasts, /experiments, and /scenarios (mode
// prop distinguishes them; /experiments also picks a specific experiment
// kind via the URL). Combines the autosaving draft form (hooks/use-
// draft.ts), a data-quality preview mutation, idempotent job submission,
// and components/run-viewer.tsx for the result. See components/
// forecast-form.tsx and components/experiment-editors.tsx for the fields.
const experiments: { kind: JobKind; label: string; description: string }[] = [
  {
    kind: "backtest",
    label: "Rolling backtest",
    description:
      "Evaluate repeated forecast origins on historical observations.",
  },
  {
    kind: "joint_independent",
    label: "Joint vs independent",
    description:
      "Measure the difference between joint and independent variates.",
  },
  {
    kind: "covariates",
    label: "Covariate ablation",
    description: "Compare selected covariates on the same evaluation windows.",
  },
  {
    kind: "settings",
    label: "Configuration comparison",
    description:
      "Compare named inference settings on consistent evaluation windows.",
  },
  {
    kind: "baselines",
    label: "Baseline comparison",
    description:
      "Benchmark against classical and seasonal forecasting baselines.",
  },
  {
    kind: "anomaly",
    label: "Anomaly analysis",
    description:
      "Inspect prediction errors and unusual historical observations.",
  },
];
// Derives (or reuses) an Idempotency-Key for job submission: if the same
// workspace+draft+kind+spec was already submitted with a stored key (e.g.
// a retried click after a network hiccup), reuse that key so the server
// treats it as the same request instead of creating a duplicate job. Any
// change to the spec's content produces a new key.
function submissionKey(
  workspace: string,
  draft: string | null,
  kind: JobKind,
  spec: Spec,
) {
  const storageKey = `timesfm:submission:${workspace}:${draft ?? "new"}`;
  const fingerprint = json({ kind, spec });
  let prior: { fingerprint: string; key: string } | null = null;
  try {
    prior = JSON.parse(sessionStorage.getItem(storageKey) ?? "null");
  } catch {
    /* Recover by creating a new key when no valid submission was stored. */
  }
  const key =
    prior?.fingerprint === fingerprint ? prior.key : crypto.randomUUID();
  sessionStorage.setItem(storageKey, json({ fingerprint, key }));
  return { key, storageKey };
}

export function ForecastsPage({
  mode = "forecast",
}: {
  mode?: "forecast" | "experiment" | "scenario";
}) {
  const context = useAnalyticalContext();
  const client = useQueryClient();
  const draft = useDraft();
  const savedDrafts = useRecords("drafts");
  const selectedRun = useRun(context.runId);
  const experiment =
    experiments.find((item) => item.kind === context.experiment)?.kind ??
    "backtest";
  const kind = mode === "experiment" ? experiment : mode;
  const [validation, setValidation] = useState<string | null>(null);
  const [previewSnapshot, setPreviewSnapshot] = useState("");
  const values = draft.values;
  const snapshot = JSON.stringify(values);
  const preview = useMutation({
    mutationFn: (spec: Spec) =>
      api<Preview>(
        `/preview?workspace_id=${encodeURIComponent(context.workspace)}`,
        { method: "POST", body: json(spec) },
      ),
    onSuccess: (_, variables) => setPreviewSnapshot(json(variables)),
  });
  // Gate for job submission: the preview must match the form's current
  // values exactly (previewSnapshot === snapshot — any edit since previewing
  // invalidates it) and must have no unresolved blocking quality issues.
  // See components/input-preview.tsx for how the same "blocked" count is
  // surfaced to the user.
  const previewCurrent =
    previewSnapshot === snapshot &&
    !!preview.data &&
    !preview.data.quality.some(
      (row) => row.status === "blocked" && !row.excluded,
    );
  const job = useQuery({
    queryKey: ["job", context.jobId],
    queryFn: ({ signal }) => api<Job>(`/jobs/${context.jobId}`, { signal }),
    enabled: !!context.jobId,
    refetchInterval: 2000,
  });
  // job.data changes on every poll tick, so this ref (rather than a
  // dependency check alone) prevents re-navigating/re-invalidating every
  // 2s while a completed job's result stays open — it only fires once per
  // job id when a result first becomes available.
  const openedResult = useRef("");
  useEffect(() => {
    if (job.data?.result_id && openedResult.current !== job.data.id) {
      openedResult.current = job.data.id;
      void client.invalidateQueries({ queryKey: ["runs"] });
      context.update({
        run: job.data.result_id,
        dataset: null,
        target: null,
        variant: null,
      });
    }
  }, [job.data, client, context]);
  const submit = useMutation({
    mutationFn: async (spec: Spec) => {
      const { key, storageKey } = submissionKey(
        context.workspace,
        context.draftId,
        kind,
        spec,
      );
      const submission: JobSubmission = {
        workspace_id: context.workspace,
        kind,
        spec: { ...spec, kind },
      };
      const result = await api<Job>("/jobs", {
        method: "POST",
        headers: { "Idempotency-Key": key },
        body: json(submission),
      });
      sessionStorage.removeItem(storageKey);
      return result;
    },
    onSuccess: (result) => {
      context.update({ job: result.id });
      client.setQueryData(["job", result.id], result);
      void client.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
  const previewData = () => {
    const spec = draft.form.getValues();
    const error = schemaMessage(spec);
    setValidation(error);
    if (!error) preview.mutate(structuredClone(spec));
  };
  const run = (spec: Spec) => {
    if (!previewCurrent) {
      setValidation(
        "Preview the current configuration before submitting a job.",
      );
      return;
    }
    if (kind === "settings" && spec.configurations.length < 2) {
      setValidation("Add at least two named configurations to compare.");
      return;
    }
    if (kind === "scenario" && !spec.scenarios.length) {
      setValidation("Add at least one scenario.");
      return;
    }
    if (kind === "covariates" && !spec.analysis.selected_covariates.length) {
      setValidation("Select at least one covariate to assess.");
      return;
    }
    setValidation(null);
    submit.mutate(structuredClone(spec));
  };
  // Changing dataset versions starts a fresh spec for those versions
  // (column roles/mapping don't carry over, since a new dataset's columns
  // may not match), while explicitly preserving settings and model choice
  // — selected covariates are cleared since they reference column names
  // that may no longer be valid.
  const versionsChange = (ids: string[]) => {
    const preserved = draft.form.getValues();
    const next = {
      ...defaultSpec(ids),
      settings: structuredClone(preserved.settings),
      model: structuredClone(preserved.model),
      analysis: { ...preserved.analysis, selected_covariates: [] },
    };
    sessionStorage.setItem(
      `timesfm:draft:${context.workspace}:new:${ids.join(",")}`,
      json({ spec: next, revision: null }),
    );
    draft.form.reset(next);
    context.update({ versions: ids.join(","), version: null, draft: null });
    setPreviewSnapshot("");
  };
  const copyRun = () => {
    const source = selectedRun.data?.payload.spec;
    if (!source) return;
    const initial = { ...defaultSpec(), ...source } as Spec;
    // Server-only lineage belongs to submitted runs, not editable draft contracts.
    const clean = Object.fromEntries(
      Object.keys(defaultSpec()).map((key) => [
        key,
        initial[key as keyof Spec],
      ]),
    ) as Spec;
    sessionStorage.setItem(
      `timesfm:draft:${context.workspace}:new:${clean.dataset_version_ids.join(",")}`,
      json({ spec: clean, revision: null }),
    );
    draft.form.reset(clean);
    context.update({
      versions: clean.dataset_version_ids.join(","),
      version: null,
      draft: null,
    });
  };
  const title =
    mode === "experiment"
      ? "Experiments"
      : mode === "scenario"
        ? "Scenarios"
        : "Forecasts";
  const description =
    mode === "experiment"
      ? "Test an assumption, compare methods, and inspect the evidence."
      : mode === "scenario"
        ? "Explore how future covariates change your forecast."
        : "Configure a forecast and explore the result in one workspace.";
  return (
    <>
      <PageHeading
        title={title}
        description={description}
        actions={
          <>
            <Drawer
              title="Run details"
              description="Saved configuration, provenance, and export."
              trigger={
                <Button variant="outline" className="metadata-trigger">
                  <Info />
                  View run details
                </Button>
              }
            >
              <RunMetadata run={selectedRun.data} />
            </Drawer>
            {selectedRun.data && (
              <Button variant="outline" onClick={copyRun}>
                <Copy />
                Copy run settings
              </Button>
            )}
          </>
        }
      />
      {mode === "experiment" && (
        <div className="mb-5 flex flex-wrap items-center gap-3 rounded-lg border bg-card px-4 py-3">
          <SlidersHorizontal size={16} className="text-primary" />
          <Select
            className="w-auto min-w-52"
            aria-label="Experiment type"
            disabled={draft.isLoading}
            value={kind}
            onChange={(event) =>
              context.update({ experiment: event.currentTarget.value })
            }
          >
            {experiments.map((item) => (
              <option key={item.kind} value={item.kind}>
                {item.label}
              </option>
            ))}
          </Select>
          <span className="text-caption text-muted-foreground">
            {experiments.find((item) => item.kind === kind)?.description}
          </span>
        </div>
      )}
      <FormProvider {...draft.form}>
        <form
          onSubmit={draft.form.handleSubmit(run, () =>
            setValidation(schemaMessage(draft.form.getValues())),
          )}
        >
          <div className="forecast-grid">
            <section className="panel configuration-panel">
              <div className="flex items-center justify-between border-b px-4 py-3">
                <h2 className="text-sm font-semibold">Configuration</h2>
                <Badge
                  tone={
                    draft.state === "conflict"
                      ? "warn"
                      : draft.state === "saved"
                        ? "good"
                        : "neutral"
                  }
                >
                  {draft.state === "saved" && <Check size={10} />}
                  {draft.state === "unsaved"
                    ? "Unsaved"
                    : draft.state === "saving"
                      ? "Saving…"
                      : draft.state === "conflict"
                        ? "Conflict"
                        : draft.state === "error"
                          ? "Save failed"
                          : "Saved"}
                </Badge>
              </div>
              <div className="border-b p-4">
                <Select
                  aria-label="Saved draft"
                  value={context.draftId ?? ""}
                  onChange={(event) =>
                    context.update({ draft: event.target.value || null })
                  }
                >
                  <option value="">New configuration</option>
                  {(savedDrafts.data ?? []).map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name}
                    </option>
                  ))}
                </Select>
              </div>
              {draft.state === "conflict" && (
                <div className="space-y-3 border-b bg-amber-500/5 p-4">
                  <p className="text-[11px] leading-5">
                    Another tab saved this draft. Your inputs are preserved;
                    choose how to continue.
                  </p>
                  <Button
                    type="button"
                    size="sm"
                    className="w-full"
                    onClick={() => void draft.save(true)}
                  >
                    <Copy />
                    Save as a new draft
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    className="w-full"
                    onClick={() =>
                      void draft
                        .reload()
                        .catch((error) => setValidation(error.message))
                    }
                  >
                    <RotateCcw />
                    Load current server draft
                  </Button>
                </div>
              )}
              {draft.error && (
                <div className="p-3">
                  <ErrorNotice error={draft.error} />
                  {draft.state === "error" && (
                    <Button
                      className="mt-2"
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => void draft.save()}
                    >
                      <Save />
                      Retry save
                    </Button>
                  )}
                </div>
              )}
              <div className="configuration-body">
                {draft.isLoading ? (
                  <Loading label="Restoring draft…" />
                ) : (
                  <ForecastFields
                    kind={kind}
                    onVersionsChange={versionsChange}
                  />
                )}
              </div>
              <div className="configuration-actions space-y-2">
                <Button
                  type="button"
                  variant="outline"
                  className="w-full"
                  disabled={preview.isPending || draft.isLoading}
                  onClick={previewData}
                >
                  <CheckCheck />
                  {preview.isPending
                    ? "Checking data…"
                    : "Preview data quality"}
                </Button>
                <Button
                  type="submit"
                  className="w-full"
                  disabled={
                    !previewCurrent ||
                    submit.isPending ||
                    draft.state === "conflict" ||
                    draft.isLoading
                  }
                >
                  <Play />
                  {submit.isPending
                    ? "Submitting…"
                    : mode === "forecast"
                      ? "Run forecast"
                      : mode === "scenario"
                        ? "Run scenarios"
                        : "Run experiment"}
                </Button>
                <p className="text-micro text-center text-muted-foreground">
                  {previewCurrent
                    ? "Current inputs checked. Ready to submit."
                    : "Preview these inputs before starting a job."}
                </p>
              </div>
            </section>
            <div className="min-w-0 space-y-4">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={
                  ![
                    ...values.mapping.past_only,
                    ...values.mapping.past_future,
                  ].some(
                    (name) =>
                      !(values.disabled_covariates ?? []).includes(name),
                  )
                }
                onClick={async () => {
                  try {
                    const saved = await api<{ id: string }>("/drafts", {
                      method: "POST",
                      body: json({
                        workspace_id: context.workspace,
                        name: "Signal usefulness",
                        payload: { spec: values },
                      }),
                    });
                    context.update(
                      {
                        experiment: "covariates",
                        draft: saved.id,
                        job: null,
                        run: null,
                      },
                      "/experiments",
                    );
                  } catch (error) {
                    setValidation(
                      error instanceof Error
                        ? error.message
                        : "Could not create comparison draft.",
                    );
                  }
                }}
              >
                Compare signal usefulness
              </Button>
              <ErrorNotice
                error={validation || submit.error || preview.error || job.error}
              />
              {context.jobId && job.data && (
                <div className="panel p-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <JobStatus job={job.data} />
                      <span className="text-sm">
                        {job.data.stage || "Waiting for a worker"}
                      </span>
                    </div>
                    <JobActions job={job.data} />
                  </div>
                  {job.data.error && (
                    <div className="mt-3">
                      <ErrorNotice error={job.data.error} />
                    </div>
                  )}
                  <p className="text-micro mt-2 text-muted-foreground">
                    Attempt {job.data.attempt} · {job.data.id} · You can leave
                    this page while it runs.
                  </p>
                </div>
              )}
              {preview.data && (
                <InputPreview preview={preview.data} current={previewCurrent} />
              )}
              {kind === "settings" && <ConfigurationEditor />}
              {kind === "scenario" && (
                <ScenarioEditor
                  template={preview.data?.scenario_template ?? []}
                />
              )}
              <RunViewer />
            </div>
            <aside className="forecast-metadata panel">
              <div className="flex items-center gap-2 border-b px-4 py-3 text-sm font-semibold">
                <FileClock size={15} />
                Run details
              </div>
              <RunMetadata run={selectedRun.data} />
            </aside>
          </div>
        </form>
      </FormProvider>
    </>
  );
}
