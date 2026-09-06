"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { History, Save, Settings2, Trash2 } from "lucide-react";
import { api, json } from "@/lib/api";
import type { Row } from "@/lib/types";
import { dateLabel } from "@/lib/utils";
import { useAnalyticalContext } from "@/hooks/use-context";
import { useRecords } from "@/hooks/use-records";
import { Button } from "./ui/button";
import {
  Badge,
  Check,
  ErrorNotice,
  Field,
  Input,
  Loading,
} from "./ui/controls";
import { Drawer } from "./ui/dialog";
import { DataTable } from "./data-table";

type Policy = {
  enabled: boolean;
  max_age_days: number | null;
  max_runs: number | null;
};
type CleanupPreview = {
  enabled: boolean;
  candidates: Row[];
  protected: Row[];
  removed?: string[];
  retained?: string[];
};

function SettingsContent() {
  const { workspace } = useAnalyticalContext();
  const client = useQueryClient();
  const path = `/workspaces/${encodeURIComponent(workspace)}/retention`;
  const policy = useQuery({
    queryKey: ["retention", workspace],
    queryFn: ({ signal }) => api<Policy>(path, { signal }),
  });
  const activity = useRecords("activity");
  const [edited, setEdited] = useState<Policy | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [notice, setNotice] = useState("");
  const values = edited ?? policy.data;
  const preview = useMutation({
    mutationFn: () => api<CleanupPreview>(`${path}/preview`),
    onSuccess: () => setConfirmed(false),
  });
  const save = useMutation({
    mutationFn: (value: Policy) =>
      api<Policy>(path, { method: "PATCH", body: json(value) }),
    onSuccess: (result) => {
      client.setQueryData(["retention", workspace], result);
      setEdited(null);
      preview.reset();
      setConfirmed(false);
      setNotice("Retention policy saved.");
    },
  });
  const apply = useMutation({
    mutationFn: () => api<CleanupPreview>(`${path}/apply`, { method: "POST" }),
    onSuccess: (result) => {
      setNotice(
        `Removed ${result.removed?.length ?? 0} runs. Protected or newly referenced runs were retained.`,
      );
      setConfirmed(false);
      preview.reset();
      void client.invalidateQueries({ queryKey: ["runs"] });
      void client.invalidateQueries({ queryKey: ["activity"] });
    },
  });
  return (
    <div className="space-y-7">
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold">Run retention</h2>
          <Badge tone={values?.enabled ? "warn" : "neutral"}>
            {values?.enabled ? "Enabled" : "Disabled"}
          </Badge>
        </div>
        <p className="mb-4 text-xs leading-6 text-muted-foreground">
          Retention is off by default. A saved policy chooses eligible runs;
          cleanup is an explicit action. Tracked runs and referenced artifacts
          remain protected.
        </p>
        <ErrorNotice error={policy.error} />
        {policy.isPending ? (
          <Loading />
        ) : (
          values && (
            <form
              onSubmit={(event) => {
                event.preventDefault();
                save.mutate(values);
              }}
              className="space-y-4"
            >
              <Check
                label="Enable a retention policy"
                checked={values.enabled}
                onChange={(event) => {
                  setEdited({ ...values, enabled: event.target.checked });
                  setConfirmed(false);
                }}
              />
              <div className="grid grid-cols-2 gap-3">
                <Field
                  label="Maximum age (days)"
                  hint="Blank means no age limit."
                >
                  <Input
                    type="number"
                    min={1}
                    value={values.max_age_days ?? ""}
                    onChange={(event) => {
                      setEdited({
                        ...values,
                        max_age_days: event.target.value
                          ? Number(event.target.value)
                          : null,
                      });
                      setConfirmed(false);
                    }}
                  />
                </Field>
                <Field
                  label="Maximum saved runs"
                  hint="Blank means no count limit."
                >
                  <Input
                    type="number"
                    min={1}
                    value={values.max_runs ?? ""}
                    onChange={(event) => {
                      setEdited({
                        ...values,
                        max_runs: event.target.value
                          ? Number(event.target.value)
                          : null,
                      });
                      setConfirmed(false);
                    }}
                  />
                </Field>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="submit"
                  variant="outline"
                  disabled={save.isPending}
                >
                  <Save />
                  Save policy
                </Button>
                <Button
                  variant="outline"
                  disabled={!!edited || !values.enabled || preview.isPending}
                  onClick={() => preview.mutate()}
                >
                  Preview cleanup
                </Button>
              </div>
            </form>
          )
        )}
        <div className="mt-3">
          <ErrorNotice error={save.error || preview.error || apply.error} />
        </div>
        {notice && (
          <p role="status" className="mt-3 text-xs leading-5 text-primary">
            {notice}
          </p>
        )}
        {preview.data && (
          <div className="mt-4 space-y-3 rounded-md border p-3">
            <p className="text-xs font-medium">
              {preview.data.candidates.length} eligible ·{" "}
              {preview.data.protected.length} protected
            </p>
            {preview.data.candidates.length > 0 && (
              <>
                <DataTable
                  data={preview.data.candidates.map(
                    ({ id, name, created_at, reasons }) => ({
                      id,
                      name,
                      created_at,
                      reasons,
                    }),
                  )}
                  caption="Runs eligible for retention cleanup"
                />
                <Check
                  label="Apply this policy and remove eligible saved runs"
                  checked={confirmed}
                  onChange={(event) => setConfirmed(event.target.checked)}
                />
                <Button
                  variant="destructive"
                  disabled={!confirmed || !!edited || apply.isPending}
                  onClick={() => apply.mutate()}
                >
                  <Trash2 />
                  Apply retention policy
                </Button>
              </>
            )}
            {preview.data.protected.length > 0 && (
              <details>
                <summary className="cursor-pointer text-[11px]">
                  Protected runs and reasons
                </summary>
                <DataTable
                  data={preview.data.protected}
                  caption="Protected runs"
                />
              </details>
            )}
          </div>
        )}
      </section>
      <section className="border-t pt-5">
        <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold">
          <History size={15} />
          Workspace activity
        </h2>
        <ErrorNotice error={activity.error} />
        {activity.isPending ? (
          <Loading />
        ) : activity.data?.length ? (
          <div className="max-h-80 divide-y overflow-auto">
            {activity.data.slice(0, 40).map((item) => (
              <details key={item.id} className="py-3">
                <summary className="cursor-pointer text-xs">
                  <span className="font-medium">{item.name}</span>
                  <span className="mt-1 block text-[10px] text-muted-foreground">
                    {dateLabel(item.created_at)}
                  </span>
                </summary>
                <pre className="mt-2 whitespace-pre-wrap break-all rounded-md bg-muted p-2 font-mono text-[10px] leading-5">
                  {JSON.stringify(item.payload, null, 2)}
                </pre>
              </details>
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">
            Recorded workspace actions appear here.
          </p>
        )}
      </section>
    </div>
  );
}

export function WorkspaceSettings() {
  return (
    <Drawer
      title="Workspace settings"
      description="Retention controls and the history of local workspace actions."
      trigger={
        <Button variant="ghost" className="w-full justify-start text-[11px]">
          <Settings2 />
          Workspace settings
        </Button>
      }
    >
      <SettingsContent />
    </Drawer>
  );
}
