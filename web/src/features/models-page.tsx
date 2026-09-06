"use client";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Box, CheckCheck, Cpu, Folder, Plus } from "lucide-react";
import { api, json } from "@/lib/api";
import type { Job, RecordItem } from "@/lib/types";
import { useAnalyticalContext } from "@/hooks/use-context";
import { useJobs, useRecords } from "@/hooks/use-records";
import { dateLabel } from "@/lib/utils";
import { JobList } from "@/components/jobs";
import { WorkerSummary } from "@/components/worker-summary";
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

const modelSchema = z.object({
  name: z.string().min(1, "Enter a model name."),
  kind: z.enum(["hub", "local"]),
  source: z.string().min(1, "Enter a model repository or directory."),
  revision: z.string(),
  offline: z.boolean(),
});
type ModelForm = z.infer<typeof modelSchema>;
type ModelRecord = RecordItem<{
  kind: string;
  source: string;
  revision: string | null;
  offline: boolean;
}>;
export function ModelsPage() {
  const context = useAnalyticalContext();
  const client = useQueryClient();
  const models = useRecords<ModelRecord>("models");
  const jobs = useJobs();
  const [notice, setNotice] = useState("");
  const form = useForm<ModelForm>({
    resolver: zodResolver(modelSchema),
    defaultValues: {
      name: "TimesFM-3",
      kind: "hub",
      source: "google/timesfm-3.0-pytorch",
      revision: "",
      offline: false,
    },
  });
  const save = useMutation({
    mutationFn: (values: ModelForm) =>
      api<ModelRecord>("/models", {
        method: "POST",
        body: json({
          workspace_id: context.workspace,
          name: values.name,
          payload: {
            kind: values.kind,
            source: values.source,
            revision: values.revision || null,
            offline: values.offline,
          },
        }),
      }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["models"] });
      setNotice(
        "Model registered. It is now available in forecast configurations.",
      );
    },
  });
  const check = useMutation({
    mutationFn: (id: string) =>
      api<Job>(`/models/${id}/check`, { method: "POST" }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["jobs"] });
      setNotice("Model check queued. Follow its progress below.");
    },
  });
  return (
    <>
      <PageHeading
        title="Models"
        description="Register checkpoints, pin revisions, and verify availability on your worker."
      />
      <div className="grid items-start gap-6 xl:grid-cols-[380px_minmax(0,1fr)]">
        <Section
          title="Register a checkpoint"
          description="Registration saves a reference; checking it is an explicit job."
        >
          <form
            onSubmit={form.handleSubmit((values) => save.mutate(values))}
            className="space-y-4 p-4"
          >
            <Field label="Display name">
              <Input {...form.register("name")} />
            </Field>
            <Field label="Source type">
              <Select {...form.register("kind")}>
                <option value="hub">Hugging Face Hub</option>
                <option value="local">Local checkpoint directory</option>
              </Select>
            </Field>
            <Field
              label={
                form.watch("kind") === "hub"
                  ? "Repository"
                  : "Directory on the worker"
              }
            >
              <Input {...form.register("source")} />
            </Field>
            <Field
              label="Revision"
              hint="Use a commit hash to pin a Hub checkpoint."
            >
              <Input
                {...form.register("revision")}
                placeholder="Default revision"
              />
            </Field>
            <Check
              label="Offline / cached files only"
              {...form.register("offline")}
            />
            <ErrorNotice
              error={
                Object.values(form.formState.errors)
                  .map((item) => item.message)
                  .filter(Boolean)
                  .join(" ") || save.error
              }
            />
            <Button type="submit" className="w-full" disabled={save.isPending}>
              <Plus />
              {save.isPending ? "Registering…" : "Register model"}
            </Button>
          </form>
        </Section>
        <div className="space-y-5">
          <WorkerSummary />
          {notice && (
            <div
              role="status"
              className="rounded-md border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-xs"
            >
              {notice}
            </div>
          )}
          <ErrorNotice error={check.error} />
          <Section title="Registered models">
            <ErrorNotice error={models.error} />
            {models.isPending ? (
              <Loading />
            ) : !models.data?.length ? (
              <Empty
                icon={<Box size={24} />}
                title="Your checkpoint registry"
                description="Register the default TimesFM-3 checkpoint or a local fine-tuned model to reuse it across experiments."
              />
            ) : (
              <div className="divide-y">
                {models.data.map((model) => (
                  <div key={model.id} className="p-4">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="flex items-start gap-3">
                        <div className="rounded-md bg-primary/5 p-2 text-primary">
                          {model.payload.kind === "local" ? (
                            <Folder size={19} />
                          ) : (
                            <Box size={19} />
                          )}
                        </div>
                        <div>
                          <h3 className="text-xs font-semibold">
                            {model.name}
                          </h3>
                          <p className="mt-1 break-all font-mono text-[10px] text-muted-foreground">
                            {model.payload.source}
                          </p>
                        </div>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={check.isPending}
                        onClick={() => check.mutate(model.id)}
                      >
                        <CheckCheck />
                        Check model
                      </Button>
                    </div>
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <Badge>
                        {model.payload.kind === "hub"
                          ? "Hub checkpoint"
                          : "Local directory"}
                      </Badge>
                      {model.payload.offline && (
                        <Badge tone="blue">Offline</Badge>
                      )}
                      <span className="text-[10px] text-muted-foreground">
                        Revision: {model.payload.revision || "default"} ·{" "}
                        {dateLabel(model.created_at)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Section>
          <Section
            title="Model checks"
            description="Check results are persistent jobs and include worker errors."
          >
            <ErrorNotice error={jobs.error} />
            <JobList
              jobs={(jobs.data ?? []).filter(
                (job) => job.kind === "model_check",
              )}
              limit={6}
            />
          </Section>
          <div className="flex items-start gap-3 rounded-lg border bg-card p-4">
            <Cpu size={18} className="mt-0.5 shrink-0 text-primary" />
            <p className="text-[11px] leading-6 text-muted-foreground">
              Checkpoints load in the native worker process. Forecast results
              record the resolved model provenance so future runs can be
              compared against the same weights.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
