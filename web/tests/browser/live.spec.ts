import { expect, test } from "@playwright/test";
import type { Run, TableData } from "../../src/lib/types";

test("existing native GPU results and data preview work through the Next proxy", async ({
  page,
  request,
}) => {
  test.skip(
    process.env.TIMESFM_LIVE_TEST !== "1",
    "Opt in with TIMESFM_LIVE_TEST=1 while the native API has a completed forecast.",
  );
  const health = await request.get("/api/v1/health");
  expect(health.ok()).toBeTruthy();
  const response = await request.get("/api/v1/runs");
  expect(response.ok()).toBeTruthy();
  const runs: Run[] = await response.json();
  const run = runs.find(
    (item) =>
      item.payload.kind === "forecast" &&
      (item.payload.spec.dataset_version_ids as string[] | undefined)?.length,
  );
  expect(run, "A completed native forecast is required").toBeTruthy();
  const versions = run!.payload.spec.dataset_version_ids as string[];
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`/data?versions=${versions.join(",")}`);
  await expect(
    page.getByRole("region", { name: "Dataset preview", exact: true }),
  ).toBeVisible();
  await page.goto(`/forecasts?run=${run!.id}&versions=${versions.join(",")}`);
  await expect(page.getByRole("figure").getByRole("img")).toBeVisible();
  await expect(
    page.getByRole("combobox", { name: "Result table" }),
  ).toBeVisible();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: "test-results/native-forecast-light.png",
    fullPage: false,
  });
  await page.getByLabel("Color theme").selectOption("dark");
  await page.screenshot({
    path: "test-results/native-forecast-dark.png",
    fullPage: false,
  });
  await page.getByRole("button", { name: "Use run settings" }).click();
  await page.getByRole("button", { name: "Preview data quality" }).click();
  await expect(
    page.getByRole("heading", { name: "Data quality preview" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Run forecast", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Prepared data", exact: true })
    .click();
  await expect(
    page.getByRole("region", { name: "Prepared dataset rows", exact: true }),
  ).toBeVisible();
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.getByRole("button", { name: "Details", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("button", { name: "Close details" }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});

test("native tracking uses canonical source identities and workspace settings load", async ({
  page,
  request,
}) => {
  test.skip(
    process.env.TIMESFM_LIVE_TEST !== "1",
    "Opt in against the native API.",
  );
  const runs: Run[] = await (await request.get("/api/v1/runs")).json();
  const run = runs.find(
    (item) =>
      item.payload.kind === "forecast" &&
      (item.payload.spec.dataset_version_ids as string[] | undefined)?.length &&
      Array.isArray(item.payload.manifest.datasets),
  );
  expect(run).toBeTruthy();
  const versions = run!.payload.spec.dataset_version_ids as string[];
  const created = await request.post("/api/v1/tracking", {
    data: {
      workspace_id: "local",
      name: "Browser mapping verification",
      payload: { run_id: run!.id, auto_refresh: false, associations: {} },
    },
  });
  expect(created.ok()).toBeTruthy();
  const tracked: { id: string } = await created.json();
  try {
    await page.goto(`/tracking?versions=${versions.join(",")}`);
    await page
      .getByRole("button", { name: /Browser mapping verification/ })
      .click();
    const mappings = page.getByRole("combobox", { name: /^Issued dataset:/ });
    await expect(mappings.first()).not.toHaveValue("");
    const firstIdentity = await mappings.first().inputValue();
    expect(versions).not.toContain(firstIdentity);
    await expect(
      page.getByRole("button", { name: "Evaluate against actuals" }),
    ).toBeEnabled();
    const refresh = page.getByRole("checkbox", {
      name: /^Automatically refresh this forecast/,
    });
    await expect(refresh).not.toBeChecked();
    await refresh.check();
    await expect(refresh).toBeEnabled();
    await expect(
      page.getByText("Automatic refresh enabled", { exact: true }),
    ).toBeVisible();
    await refresh.uncheck();
    await expect(refresh).toBeEnabled();
    await expect(
      page.getByText("Manual refresh only", { exact: true }),
    ).toBeVisible();
    for (const mapping of await mappings.all()) await mapping.selectOption("");
    await expect(
      page.getByRole("button", { name: "Evaluate against actuals" }),
    ).toBeDisabled();
    await mappings.first().selectOption(firstIdentity);
    await expect(
      page.getByRole("button", { name: "Evaluate against actuals" }),
    ).toBeEnabled();
    await page
      .getByRole("button", { name: "Workspace settings", exact: true })
      .click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expect(
      page.getByLabel("Maximum age (days)", { exact: false }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Workspace activity" }),
    ).toBeVisible();
    await page.screenshot({
      path: "test-results/native-workspace-settings.png",
      fullPage: false,
    });
  } finally {
    const removed = await request.delete(`/api/v1/tracking/${tracked.id}`);
    expect(removed.ok()).toBeTruthy();
  }
});

test("native calibration uses persisted coverage filtered to one horizon step", async ({
  page,
  request,
}) => {
  test.skip(
    process.env.TIMESFM_LIVE_TEST !== "1",
    "Opt in against the native API.",
  );
  const runs: Run[] = await (await request.get("/api/v1/runs")).json();
  const run = runs.find(
    (item) =>
      item.payload.kind === "backtest" && "calibration" in item.payload.tables,
  );
  expect(run, "A completed backtest with calibration is required").toBeTruthy();
  const versions = run!.payload.spec.dataset_version_ids as string[];
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/overview");
  await expect(
    page.getByRole("region", { name: "Latest saved accuracy metrics" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Active tracking" }),
  ).toBeVisible();
  await page.screenshot({
    path: "test-results/native-overview.png",
    fullPage: false,
  });
  await page.goto(`/experiments?run=${run!.id}&versions=${versions.join(",")}`);
  const firstResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return (
      url.pathname.endsWith("/tables/calibration") &&
      url.searchParams.get("step") === "1"
    );
  });
  await page.getByRole("button", { name: "Inspect coverage" }).click();
  const initial: TableData = await (await firstResponse).json();
  expect(initial.rows.length).toBeGreaterThan(0);
  expect(initial.rows.every((row) => row.step === 1)).toBe(true);
  await expect(
    page.getByRole("img", {
      name: /Observed coverage against nominal interval coverage/,
    }),
  ).toBeVisible();
  const secondResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return (
      url.pathname.endsWith("/tables/calibration") &&
      url.searchParams.get("step") === "2"
    );
  });
  await page.getByRole("spinbutton", { name: "Horizon step" }).fill("2");
  const filtered: TableData = await (await secondResponse).json();
  expect(filtered.rows.length).toBeGreaterThan(0);
  expect(filtered.rows.every((row) => row.step === 2)).toBe(true);
  await page
    .getByRole("img", {
      name: /Observed coverage against nominal interval coverage/,
    })
    .screenshot({ path: "test-results/native-calibration.png" });
  expect(errors).toEqual([]);
});
