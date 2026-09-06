import { describe, expect, it } from "vitest";
import { defaultSpec, roleAvailable, specSchema } from "../../src/lib/spec";

function valid() {
  const spec = defaultSpec(["version-a"]);
  spec.mapping.targets = ["demand"];
  return spec;
}
describe("forecast submission contract", () => {
  it("preserves numerical defaults and accepts a complete minimal mapping", () => {
    const spec = valid();
    expect(specSchema.parse(spec).settings).toMatchObject({
      horizon: 32,
      context_length: 512,
      batch_size: 4,
      mode: "multivariate",
      padding_mode: "none",
      return_quantiles: true,
    });
  });
  it("rejects incomplete drafts for submission", () => {
    expect(specSchema.safeParse(defaultSpec()).success).toBe(false);
  });
  it("prevents group identifiers from also becoming target or covariate columns", () => {
    const spec = valid();
    spec.preparation.group_columns = ["demand"];
    expect(specSchema.safeParse(spec).success).toBe(false);
    expect(roleAvailable("demand", [], spec)).toBe(false);
  });
  it("prevents timestamp and covariate overlap", () => {
    const spec = valid();
    spec.mapping.timestamp = "date";
    spec.mapping.past_future = ["date"];
    expect(specSchema.safeParse(spec).success).toBe(false);
  });
  it("uses the model's actual maximum context limit", () => {
    const spec = valid();
    spec.settings.horizon = 15361;
    expect(specSchema.safeParse(spec).success).toBe(false);
  });
  it("rejects non-finite scenario edits", () => {
    const spec = valid();
    spec.scenarios = [
      {
        name: "Promotion",
        overrides: [
          { dataset: "store", row: 9, covariate: "price", value: NaN },
        ],
      },
    ];
    expect(specSchema.safeParse(spec).success).toBe(false);
  });
});
