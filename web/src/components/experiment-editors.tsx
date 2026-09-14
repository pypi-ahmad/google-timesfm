"use client";
import { useState } from "react";
import { useFormContext, useWatch } from "react-hook-form";
import { Plus, Trash2 } from "lucide-react";
import type { Spec } from "@/lib/spec";
import type { Row } from "@/lib/types";
import { display } from "@/lib/utils";
import { Button } from "./ui/button";
import { Empty, Field, Input, Section, Select } from "./ui/controls";
import { SettingsFields } from "./forecast-form";

// Two form-connected editors used only for the "experiment"/"scenario"
// job kinds in features/forecasts-page.tsx: ConfigurationEditor compares
// up to 8 named inference-setting configurations (each reusing the
// SettingsFields from forecast-form.tsx), and ScenarioEditor edits
// known-future covariate overrides for up to 3 what-if scenarios. The
// 8/3 limits mirror lib/spec.ts's configurations/scenarios array caps.
export function ConfigurationEditor() {
  const { control, register, setValue } = useFormContext<Spec>();
  const spec = useWatch({ control }) as Spec;
  return (
    <Section
      title="Compare configurations"
      description="Each configuration records a complete set of inference settings."
      actions={
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={spec.configurations.length >= 8}
          onClick={() =>
            setValue(
              "configurations",
              [
                ...spec.configurations,
                {
                  name: `Configuration ${spec.configurations.length + 1}`,
                  settings: structuredClone(spec.settings),
                },
              ],
              { shouldDirty: true },
            )
          }
        >
          <Plus />
          Add
        </Button>
      }
    >
      <div className="space-y-3 p-4">
        {!spec.configurations.length && (
          <p className="text-xs leading-6 text-muted-foreground">
            Add at least two configurations, then vary their context length or
            inference options.
          </p>
        )}
        {spec.configurations.map((_, index) => (
          <details key={index} className="rounded-md border p-3">
            <summary className="cursor-pointer text-xs font-medium">
              {spec.configurations[index].name}
            </summary>
            <div className="mt-4 space-y-4">
              <div className="flex items-end gap-2">
                <Field className="flex-1" label="Configuration name">
                  <Input {...register(`configurations.${index}.name`)} />
                </Field>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  aria-label={`Remove configuration ${index + 1}`}
                  onClick={() =>
                    setValue(
                      "configurations",
                      spec.configurations.filter((_, i) => i !== index),
                      { shouldDirty: true },
                    )
                  }
                >
                  <Trash2 />
                </Button>
              </div>
              <SettingsFields
                prefix={`configurations.${index}.settings`}
                compact
              />
            </div>
          </details>
        ))}
      </div>
    </Section>
  );
}

export function ScenarioEditor({ template }: { template: Row[] }) {
  const { control, setValue, register } = useFormContext<Spec>();
  const spec = useWatch({ control }) as Spec;
  const [selected, setSelected] = useState(0);
  const [page, setPage] = useState(0);
  const index = Math.min(selected, Math.max(0, spec.scenarios.length - 1));
  const scenario = spec.scenarios[index];
  // Pagination here is a display-only window into the future-covariate
  // template; it doesn't affect which overrides exist, so edits on any
  // page are preserved when paging away and back.
  const rows = template.slice(page * 50, (page + 1) * 50);
  // Composite identity for a future-covariate cell, since no single
  // column uniquely identifies a (series, row, covariate) combination.
  const key = (row: Row) => `${row.dataset}:${row.row}:${row.covariate}`;
  const edit = (row: Row, value: string) => {
    if (!scenario) return;
    const remaining = scenario.overrides.filter(
      (item) => key(item) !== key(row),
    );
    const number = value === "" ? Number.NaN : Number(value);
    setValue(
      `scenarios.${index}.overrides`,
      // Editing a cell back to its baseline value removes the override
      // entirely, rather than storing a redundant no-op override.
      number === row.value
        ? remaining
        : [
            ...remaining,
            {
              dataset: String(row.dataset),
              row: Number(row.row),
              covariate: String(row.covariate),
              value: number,
            },
          ],
      { shouldDirty: true },
    );
  };
  return (
    <Section
      title="What-if scenarios"
      description="Change known-future covariates. Baseline inputs and historical observations stay fixed."
      actions={
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={spec.scenarios.length >= 3}
          onClick={() => {
            setValue(
              "scenarios",
              [
                ...spec.scenarios,
                {
                  name: `Scenario ${spec.scenarios.length + 1}`,
                  overrides: [],
                },
              ],
              { shouldDirty: true },
            );
            setSelected(spec.scenarios.length);
          }}
        >
          <Plus />
          Add scenario
        </Button>
      }
    >
      {!scenario ? (
        <Empty
          title="Explore a change in assumptions"
          description="Preview your data, then add a scenario and edit its future covariate values."
        />
      ) : (
        <>
          <div className="grid gap-3 border-b p-4 sm:grid-cols-2">
            <Field label="Scenario">
              <Select
                value={index}
                onChange={(event) => setSelected(Number(event.target.value))}
              >
                {spec.scenarios.map((item, i) => (
                  <option key={i} value={i}>
                    {item.name}
                  </option>
                ))}
              </Select>
            </Field>
            <div className="flex items-end gap-2">
              <Field className="flex-1" label="Scenario name">
                <Input {...register(`scenarios.${index}.name`)} />
              </Field>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                aria-label="Remove scenario"
                onClick={() =>
                  setValue(
                    "scenarios",
                    spec.scenarios.filter((_, i) => i !== index),
                    { shouldDirty: true },
                  )
                }
              >
                <Trash2 />
              </Button>
            </div>
          </div>
          {template.length ? (
            <>
              <div className="table-scroller">
                <table className="data-table">
                  <thead>
                    <tr>
                      {[
                        "Dataset",
                        "Row",
                        "Timestamp",
                        "Covariate",
                        "Baseline",
                        "Scenario value",
                      ].map((name) => (
                        <th key={name} scope="col">
                          {name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => {
                      const override = scenario.overrides.find(
                        (item) => key(item) === key(row),
                      );
                      return (
                        <tr key={key(row)}>
                          <td>{display(row.dataset)}</td>
                          <td>{display(row.row)}</td>
                          <td>{display(row.timestamp)}</td>
                          <td>{display(row.covariate)}</td>
                          <td>{display(row.value)}</td>
                          <td>
                            <Input
                              className={`h-7 w-28 ${override ? "border-primary bg-primary/5" : ""}`}
                              type="number"
                              step="any"
                              aria-label={`${scenario.name}, ${row.covariate}, row ${row.row}`}
                              value={
                                override
                                  ? Number.isFinite(override.value)
                                    ? override.value
                                    : ""
                                  : Number(row.value)
                              }
                              onChange={(event) =>
                                edit(row, event.target.value)
                              }
                            />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center justify-between border-t p-3 text-[10px] text-muted-foreground">
                <span>
                  {scenario.overrides.length} edits · {template.length} future
                  values
                </span>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    disabled={page === 0}
                    onClick={() => setPage(page - 1)}
                  >
                    Previous
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    disabled={(page + 1) * 50 >= template.length}
                    onClick={() => setPage(page + 1)}
                  >
                    Next
                  </Button>
                </div>
              </div>
            </>
          ) : (
            <Empty
              title="No future values available"
              description="Assign known-future covariates with complete future rows, then run the quality preview."
            />
          )}
        </>
      )}
    </Section>
  );
}
