"use client";
import { useFormContext, useWatch } from "react-hook-form";
import { Plus, X } from "lucide-react";
import { useDatasets, useRecords } from "@/hooks/use-records";
import { roleAvailable, type Spec } from "@/lib/spec";
import type { DatasetVersion, JobKind, RecordItem } from "@/lib/types";
import { dateLabel, shortId } from "@/lib/utils";
import { Button } from "./ui/button";
import { Badge, Check, ErrorNotice, Field, Input, Select } from "./ui/controls";

// The main "Configure a forecast" field set, plus two pieces reused by
// components/experiment-editors.tsx: ColumnPicker (role assignment
// checklist) and SettingsFields (the inference-settings block, also used
// per-configuration in ConfigurationEditor). Fields are bound to the
// react-hook-form context from hooks/use-draft.ts via useFormContext.
// See lib/spec.ts for the Spec shape and the role-exclusivity invariant
// that roleAvailable()/ColumnPicker enforce here.
type Role =
  | "mapping.targets"
  | "mapping.past_only"
  | "mapping.past_future"
  | "preparation.group_columns";
export function ColumnPicker({
  label,
  path,
  columns,
  hint,
}: {
  label: string;
  path: Role;
  columns: string[];
  hint?: string;
}) {
  const { setValue, control } = useFormContext<Spec>();
  const spec = useWatch({ control }) as Spec;
  // group_columns lives under `preparation`, not `mapping` like the other
  // roles, so its lookup can't share the `spec.mapping[...]` path below.
  const chosen =
    path === "preparation.group_columns"
      ? spec.preparation.group_columns
      : spec.mapping[
          path.split(".")[1] as "targets" | "past_only" | "past_future"
        ];
  return (
    <details className="rounded-md border bg-card">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-2.5 py-2 text-[11px]">
        <span>{label}</span>
        <Badge>{chosen.length ? `${chosen.length} selected` : "Choose"}</Badge>
      </summary>
      <div className="max-h-48 overflow-auto border-t px-2.5 py-1">
        {columns.length ? (
          columns.map((column) => (
            <Check
              key={column}
              label={column}
              checked={chosen.includes(column)}
              disabled={!roleAvailable(column, chosen, spec)}
              onChange={(event) =>
                setValue(
                  path,
                  event.target.checked
                    ? [...chosen, column]
                    : chosen.filter((item) => item !== column),
                  { shouldDirty: true },
                )
              }
            />
          ))
        ) : (
          <p className="py-3 text-[11px] text-muted-foreground">
            Choose dataset versions first.
          </p>
        )}
        {hint && (
          <p className="py-2 text-[10px] leading-5 text-muted-foreground">
            {hint}
          </p>
        )}
      </div>
      {chosen.length > 0 && (
        <p className="px-2.5 pb-2 text-[10px] leading-5 text-muted-foreground">
          {chosen.join(", ")}
        </p>
      )}
    </details>
  );
}

export function SettingsFields({
  prefix = "settings",
  compact = false,
}: {
  prefix?: "settings" | `configurations.${number}.settings`;
  compact?: boolean;
}) {
  const { register } = useFormContext<Spec>();
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <Field label="Horizon" hint="Forecast rows">
          <Input
            type="number"
            min={1}
            max={15360}
            {...register(`${prefix}.horizon`, { valueAsNumber: true })}
          />
        </Field>
        <Field label="Context" hint="Historical rows">
          <Input
            type="number"
            min={1}
            max={15360}
            {...register(`${prefix}.context_length`, { valueAsNumber: true })}
          />
        </Field>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Task">
          <Select {...register(`${prefix}.task`)}>
            <option value="forecast">Future forecast</option>
            <option value="holdout">Holdout evaluation</option>
          </Select>
        </Field>
        <Field label="Variates">
          <Select {...register(`${prefix}.mode`)}>
            <option value="multivariate">Joint</option>
            <option value="univariate">Independent</option>
          </Select>
        </Field>
      </div>
      <details open={compact || undefined}>
        <summary className="text-[11px] text-muted-foreground">
          Inference options
        </summary>
        <div className="mt-3 space-y-1">
          <Check
            label="Return quantiles"
            {...register(`${prefix}.return_quantiles`)}
          />
          <Check
            label="Symmetric averaging"
            {...register(`${prefix}.use_symmetric_averaging`)}
          />
          <Check
            label="Positive forecasts"
            {...register(`${prefix}.make_positive`)}
          />
          <Check
            label="Sort quantiles"
            {...register(`${prefix}.sort_quantiles`)}
          />
          <Check
            label="Normalize context (z-score)"
            {...register(`${prefix}.use_znorm`)}
          />
          <Check
            label="Allow benchmark chunking"
            {...register(`${prefix}.allow_benchmark_chunking`)}
          />
          <div className="grid grid-cols-2 gap-3 pt-2">
            <Field label="Padding">
              <Select {...register(`${prefix}.padding_mode`)}>
                <option value="none">None</option>
                <option value="edge">Edge</option>
              </Select>
            </Field>
            <Field label="Batch size">
              <Input
                type="number"
                min={1}
                max={64}
                {...register(`${prefix}.batch_size`, { valueAsNumber: true })}
              />
            </Field>
          </div>
        </div>
      </details>
    </div>
  );
}

export function ForecastFields({
  kind,
  onVersionsChange,
}: {
  kind: JobKind;
  onVersionsChange: (ids: string[]) => void;
}) {
  const { control, register, setValue } = useFormContext<Spec>();
  const spec = useWatch({ control }) as Spec;
  const library = useDatasets();
  const models = useRecords<RecordItem<Spec["model"]>>("models");
  const selected = library.versions.filter((item) =>
    spec.dataset_version_ids.includes(item.id),
  );
  const columns = selected.length
    ? selected[0].payload.columns.filter((column) =>
        selected.every((item) => item.payload.columns.includes(column)),
      )
    : [];
  const unassigned = columns.filter((column) =>
    roleAvailable(
      column,
      spec.mapping.timestamp ? [spec.mapping.timestamp] : [],
      spec,
    ),
  );
  // Selecting a version deselects any other version of the same
  // dataset_id first — only one version per underlying dataset can be
  // active in a spec at once.
  const changeVersion = (item: DatasetVersion, checked: boolean) => {
    const sameDataset = library.versions
      .filter(
        (version) => version.payload.dataset_id === item.payload.dataset_id,
      )
      .map((version) => version.id);
    const next = spec.dataset_version_ids.filter(
      (id) => !sameDataset.includes(id),
    );
    onVersionsChange(checked ? [...next, item.id] : next);
  };
  return (
    <>
      <div className="form-section">
        <h3>01 · Dataset versions</h3>
        <ErrorNotice error={library.error} />
        <details className="rounded-md border">
          <summary className="px-2.5 py-2 text-[11px]">
            {selected.length
              ? `${selected.length} dataset${selected.length > 1 ? "s" : ""} selected`
              : "Select dataset versions"}
          </summary>
          <div className="max-h-56 overflow-auto border-t px-2.5 py-1">
            {library.versions.map((item) => (
              <Check
                key={item.id}
                label={item.name || item.payload.filename}
                hint={`${shortId(item.id)} · ${dateLabel(item.created_at)}`}
                checked={spec.dataset_version_ids.includes(item.id)}
                onChange={(event) => changeVersion(item, event.target.checked)}
              />
            ))}
            {!library.versions.length && (
              <p className="py-3 text-[11px] text-muted-foreground">
                Upload a dataset on the Data page.
              </p>
            )}
          </div>
        </details>
        {selected.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {selected.map((item) => (
              <Badge key={item.id}>{item.name || item.payload.filename}</Badge>
            ))}
          </div>
        )}
        <p className="mt-2 text-[10px] leading-5 text-muted-foreground">
          Versions are fixed. Changing inputs starts a new draft and clears
          column roles.
        </p>
      </div>
      <div className="form-section">
        <h3>02 · Column roles</h3>
        <div className="space-y-3">
          <Field label="Timestamp">
            <Select
              value={spec.mapping.timestamp ?? ""}
              onChange={(event) =>
                setValue("mapping.timestamp", event.target.value || null, {
                  shouldDirty: true,
                })
              }
            >
              <option value="">Row order</option>
              {unassigned.map((column) => (
                <option key={column}>{column}</option>
              ))}
            </Select>
          </Field>
          <ColumnPicker
            label="Targets"
            path="mapping.targets"
            columns={columns}
            hint="The numeric series to forecast."
          />
          <ColumnPicker
            label="Past-only covariates"
            path="mapping.past_only"
            columns={columns}
          />
          <ColumnPicker
            label="Known-future covariates"
            path="mapping.past_future"
            columns={columns}
            hint="Provide complete values for the entire forecast horizon."
          />
        </div>
      </div>
      <div className="form-section">
        <h3>03 · Forecast settings</h3>
        <SettingsFields />
      </div>
      <div className="form-section">
        <details>
          <summary>Grouping and calendar</summary>
          <div className="space-y-3">
            <ColumnPicker
              label="Group identifiers"
              path="preparation.group_columns"
              columns={columns}
              hint="Each unique group becomes a separate series."
            />
            <Field
              label="Frequency"
              hint="Leave blank to infer. Examples: D, h, W-MON, MS."
            >
              <Input
                value={spec.preparation.frequency ?? ""}
                placeholder="Infer from timestamps"
                onChange={(event) =>
                  setValue(
                    "preparation.frequency",
                    event.target.value || null,
                    { shouldDirty: true },
                  )
                }
              />
            </Field>
            <Check
              label="Weekday covariate"
              {...register("preparation.calendar.weekday")}
            />
            <Check
              label="Month covariate"
              {...register("preparation.calendar.month")}
            />
            <div className="grid grid-cols-2 gap-3">
              <Field label="Holiday country">
                <Input
                  value={spec.preparation.calendar.holiday_country ?? ""}
                  placeholder="e.g. IN"
                  onChange={(event) =>
                    setValue(
                      "preparation.calendar.holiday_country",
                      event.target.value || null,
                      { shouldDirty: true },
                    )
                  }
                />
              </Field>
              <Field label="Subdivision">
                <Input
                  value={spec.preparation.calendar.holiday_subdivision ?? ""}
                  placeholder="Optional"
                  onChange={(event) =>
                    setValue(
                      "preparation.calendar.holiday_subdivision",
                      event.target.value || null,
                      { shouldDirty: true },
                    )
                  }
                />
              </Field>
            </div>
            {spec.preparation.calendar.events.map((event, index) => (
              <div key={index} className="space-y-2 rounded-md border p-2">
                <div className="flex items-center gap-1">
                  <Input
                    aria-label={`Event ${index + 1} name`}
                    placeholder="Event name"
                    {...register(`preparation.calendar.events.${index}.name`)}
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    type="button"
                    aria-label={`Remove event ${index + 1}`}
                    onClick={() =>
                      setValue(
                        "preparation.calendar.events",
                        spec.preparation.calendar.events.filter(
                          (_, i) => i !== index,
                        ),
                        { shouldDirty: true },
                      )
                    }
                  >
                    <X />
                  </Button>
                </div>
                {/* Plain YYYY-MM-DD strings, no timezone attached (lib/
                    spec.ts types these as z.string()). Whether the server
                    treats calendar event dates as UTC or dataset-local
                    isn't determined by this file; check the backend
                    calendar-feature implementation before assuming. */}
                <Field label="Start date">
                  <Input
                    type="date"
                    {...register(`preparation.calendar.events.${index}.start`)}
                  />
                </Field>
                <Field label="End date">
                  <Input
                    type="date"
                    value={event.end ?? ""}
                    onChange={(e) =>
                      setValue(
                        `preparation.calendar.events.${index}.end`,
                        e.target.value || null,
                        { shouldDirty: true },
                      )
                    }
                  />
                </Field>
              </div>
            ))}
            <Button
              variant="outline"
              size="sm"
              type="button"
              onClick={() =>
                setValue(
                  "preparation.calendar.events",
                  [
                    ...spec.preparation.calendar.events,
                    { name: "", start: "", end: null },
                  ],
                  { shouldDirty: true },
                )
              }
            >
              <Plus />
              Add calendar event
            </Button>
          </div>
        </details>
      </div>
      <div className="form-section">
        <details>
          <summary>Model selection</summary>
          <div className="space-y-3">
            {models.data?.length ? (
              <Field label="Registered model">
                <Select
                  value=""
                  onChange={(event) => {
                    const model = models.data.find(
                      (item) => item.id === event.target.value,
                    );
                    if (model)
                      setValue("model", model.payload, { shouldDirty: true });
                  }}
                >
                  <option value="">Choose a saved model…</option>
                  {models.data.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </Select>
              </Field>
            ) : null}
            <Field label="Source type">
              <Select {...register("model.kind")}>
                <option value="hub">Hugging Face Hub</option>
                <option value="local">Local directory</option>
              </Select>
            </Field>
            <Field
              label={
                spec.model.kind === "hub"
                  ? "Model repository"
                  : "Checkpoint directory"
              }
            >
              <Input {...register("model.source")} />
            </Field>
            <Field
              label="Revision"
              hint="A commit hash pins Hub weights for reproducible runs."
            >
              <Input
                value={spec.model.revision ?? ""}
                placeholder="Default revision"
                onChange={(event) =>
                  setValue("model.revision", event.target.value || null, {
                    shouldDirty: true,
                  })
                }
              />
            </Field>
            <Check
              label="Offline / cached files only"
              {...register("model.offline")}
            />
          </div>
        </details>
      </div>
      {kind !== "forecast" && kind !== "scenario" && (
        <div className="form-section">
          <h3>Experiment design</h3>
          <div className="space-y-3">
            <Field label="Rolling windows">
              <Input
                type="number"
                min={1}
                max={1000}
                {...register("analysis.windows", { valueAsNumber: true })}
              />
            </Field>
            <Field
              label="Stride"
              hint="Leave blank to use the forecast horizon."
            >
              <Input
                type="number"
                min={1}
                value={spec.analysis.stride ?? ""}
                onChange={(event) =>
                  setValue(
                    "analysis.stride",
                    event.target.value ? Number(event.target.value) : null,
                    { shouldDirty: true },
                  )
                }
              />
            </Field>
            {kind === "baselines" && (
              <Field
                label="Seasonal period"
                hint="Rows in one season, e.g. 7 for daily data with a weekly cycle."
              >
                <Input
                  type="number"
                  min={1}
                  max={15360}
                  {...register("analysis.seasonal_period", {
                    valueAsNumber: true,
                  })}
                />
              </Field>
            )}
            {kind === "covariates" && (
              <div>
                <p className="mb-2 text-[11px] font-medium">
                  Covariates to assess
                </p>
                {[...spec.mapping.past_only, ...spec.mapping.past_future].map(
                  (column) => (
                    <Check
                      key={column}
                      label={column}
                      checked={spec.analysis.selected_covariates.includes(
                        column,
                      )}
                      onChange={(event) =>
                        setValue(
                          "analysis.selected_covariates",
                          event.target.checked
                            ? [...spec.analysis.selected_covariates, column]
                            : spec.analysis.selected_covariates.filter(
                                (value) => value !== column,
                              ),
                          { shouldDirty: true },
                        )
                      }
                    />
                  ),
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
