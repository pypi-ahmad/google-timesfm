import { z } from "zod";

export const settingsSchema = z.object({
  horizon: z.number().int().min(1).max(15360),
  context_length: z.number().int().min(1).max(15360),
  task: z.enum(["forecast", "holdout"]),
  mode: z.enum(["multivariate", "univariate"]),
  return_quantiles: z.boolean(),
  use_symmetric_averaging: z.boolean(),
  make_positive: z.boolean(),
  sort_quantiles: z.boolean(),
  use_znorm: z.boolean(),
  padding_mode: z.enum(["none", "edge"]),
  batch_size: z.number().int().min(1).max(64),
  allow_benchmark_chunking: z.boolean(),
});
export const specSchema = z
  .object({
    dataset_version_ids: z
      .array(z.string())
      .min(1, "Select a dataset version."),
    mapping: z.object({
      timestamp: z.string().nullable(),
      targets: z.array(z.string()).min(1, "Select at least one target."),
      past_only: z.array(z.string()),
      past_future: z.array(z.string()),
    }),
    settings: settingsSchema,
    preparation: z.object({
      group_columns: z.array(z.string()),
      excluded_groups: z.array(z.string()),
      frequency: z.string().nullable(),
      calendar: z.object({
        weekday: z.boolean(),
        month: z.boolean(),
        holiday_country: z.string().nullable(),
        holiday_subdivision: z.string().nullable(),
        events: z.array(
          z.object({
            name: z.string(),
            start: z.string(),
            end: z.string().nullable(),
          }),
        ),
      }),
    }),
    model: z.object({
      kind: z.enum(["hub", "local"]),
      source: z.string().min(1),
      revision: z.string().nullable(),
      offline: z.boolean(),
    }),
    analysis: z.object({
      windows: z.number().int().min(1).max(1000),
      stride: z.number().int().positive().nullable(),
      seasonal_period: z.number().int().positive().max(15360),
      selected_covariates: z.array(z.string()),
    }),
    configurations: z
      .array(z.object({ name: z.string().min(1), settings: settingsSchema }))
      .max(8),
    scenarios: z
      .array(
        z.object({
          name: z.string().min(1),
          overrides: z.array(
            z.object({
              dataset: z.string(),
              row: z.number().int().nonnegative(),
              covariate: z.string(),
              value: z.number(),
            }),
          ),
        }),
      )
      .max(3),
  })
  .superRefine((spec, ctx) => {
    const roles = [
      ...spec.mapping.targets,
      ...spec.mapping.past_only,
      ...spec.mapping.past_future,
      ...spec.preparation.group_columns,
      ...(spec.mapping.timestamp ? [spec.mapping.timestamp] : []),
    ];
    if (new Set(roles).size !== roles.length)
      ctx.addIssue({
        code: "custom",
        path: ["mapping"],
        message:
          "Timestamp, groups, targets, and covariates must use different columns.",
      });
  });
export type Spec = z.infer<typeof specSchema>;
export const defaultSettings: Spec["settings"] = {
  horizon: 32,
  context_length: 512,
  task: "forecast",
  mode: "multivariate",
  return_quantiles: true,
  use_symmetric_averaging: true,
  make_positive: true,
  sort_quantiles: true,
  use_znorm: false,
  padding_mode: "none",
  batch_size: 4,
  allow_benchmark_chunking: false,
};
export function defaultSpec(versions: string[] = []): Spec {
  return {
    dataset_version_ids: versions,
    mapping: { timestamp: null, targets: [], past_only: [], past_future: [] },
    settings: { ...defaultSettings },
    preparation: {
      group_columns: [],
      excluded_groups: [],
      frequency: null,
      calendar: {
        weekday: false,
        month: false,
        holiday_country: null,
        holiday_subdivision: null,
        events: [],
      },
    },
    model: {
      kind: "hub",
      source: "google/timesfm-3.0-pytorch",
      revision: null,
      offline: false,
    },
    analysis: {
      windows: 5,
      stride: null,
      seasonal_period: 1,
      selected_covariates: [],
    },
    configurations: [],
    scenarios: [],
  };
}
export function schemaMessage(spec: unknown): string | null {
  const result = specSchema.safeParse(spec);
  return result.success
    ? null
    : result.error.issues.map((issue) => issue.message).join(" ");
}
export function roleAvailable(column: string, current: string[], spec: Spec) {
  return (
    current.includes(column) ||
    ![
      spec.mapping.timestamp,
      ...spec.mapping.targets,
      ...spec.mapping.past_only,
      ...spec.mapping.past_future,
      ...spec.preparation.group_columns,
    ].includes(column)
  );
}
