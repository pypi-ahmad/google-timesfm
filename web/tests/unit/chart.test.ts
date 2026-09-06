import { describe, expect, it } from "vitest";
import {
  bandSegments,
  crossedQuantiles,
  finiteNumber,
} from "../../src/lib/chart";

describe("nominal interval geometry", () => {
  it("retains negative bounds without zero-based stacking", () => {
    expect(
      bandSegments(
        [
          { step: 1, "q0.1": -20, "q0.9": -5 },
          { step: 2, "q0.1": -19, "q0.9": -3 },
        ],
        "q0.1",
        "q0.9",
        4,
      ),
    ).toEqual([
      [
        [4, -20, -5],
        [5, -19, -3],
      ],
    ]);
  });
  it("splits a band at missing bounds", () => {
    expect(
      bandSegments(
        [
          { "q0.1": 1, "q0.9": 4 },
          { "q0.1": null, "q0.9": 3 },
          { "q0.1": 2, "q0.9": 5 },
        ],
        "q0.1",
        "q0.9",
      ),
    ).toEqual([[[0, 1, 4]], [[2, 2, 5]]]);
  });
  it("splits a band when forecast steps have a gap", () => {
    expect(
      bandSegments(
        [
          { step: 1, "q0.1": 1, "q0.9": 4 },
          { step: 3, "q0.1": 2, "q0.9": 5 },
        ],
        "q0.1",
        "q0.9",
      ),
    ).toHaveLength(2);
  });
  it("omits every interval at a crossed quantile row without repairing predictions", () => {
    const row = { "q0.1": -3, "q0.2": 4, "q0.3": 1, "q0.9": 8 };
    expect(crossedQuantiles(row)).toBe(true);
    expect(bandSegments([row], "q0.1", "q0.9")).toEqual([]);
    expect(row["q0.2"]).toBe(4);
  });
  it("does not coerce missing, string, or infinite values into data", () => {
    expect([null, undefined, "2", Infinity, NaN].map(finiteNumber)).toEqual([
      null,
      null,
      null,
      null,
      null,
    ]);
    expect(finiteNumber(0)).toBe(0);
  });
});
