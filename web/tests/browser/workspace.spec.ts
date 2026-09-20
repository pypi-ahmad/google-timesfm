import { expect, test, type BrowserContext } from "@playwright/test";
import { defaultSpec } from "../../src/lib/spec";

const timestamp = "2026-09-06T10:00:00Z";
const spec = defaultSpec(["version-a"]);
spec.mapping = {
  timestamp: "date",
  targets: ["demand"],
  past_only: [],
  past_future: ["price"],
};
const record = <T>(id: string, kind: string, name: string, payload: T) => ({
  id,
  workspace_id: "local",
  kind,
  name,
  payload,
  revision: 1,
  created_at: timestamp,
});
const dataset = record("dataset-a", "dataset", "Retail demand", {});
const version = record("version-a", "dataset_version", "Retail demand · v1", {
  dataset_id: "dataset-a",
  filename: "demand.csv",
  source_name: "Retail demand",
  columns: ["date", "store", "demand", "price"],
  rows: 512,
  artifact: { key: "input", sha256: "123", size: 8000 },
});
const forecastRows = Array.from({ length: 125 }, (_, i) => ({
  dataset: "Retail demand",
  target: "demand",
  step: i + 1,
  timestamp: `2026-10-${String((i % 28) + 1).padStart(2, "0")}`,
  point: i - 20,
  actual: null,
  ...Object.fromEntries(
    Array.from({ length: 9 }, (_, q) => [`q0.${q + 1}`, i - 24 + q]),
  ),
}));
const run = record("run-a", "run", "Demand · holdout", {
  kind: "forecast",
  spec,
  manifest: { model_provenance: { checkpoint: "google/timesfm-3.0-pytorch" } },
  tables: {
    forecast: { key: "forecast" },
    metrics: { key: "metrics" },
    calibration: { key: "calibration" },
  },
});

async function fixture(context: BrowserContext) {
  let draft = record("draft-a", "draft", "Demand investigation", {
    spec: structuredClone(spec),
  });
  const submissions: { body: unknown; key: string | undefined }[] = [];
  const requests: string[] = [];
  await context.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname.replace("/api/v1", "");
    requests.push(url.pathname + url.search);
    const respond = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    if (path === "/health") return respond({ status: "ok" });
    if (path === "/workspaces")
      return respond([record("local", "workspace", "Local", {})]);
    if (path === "/datasets") return respond([dataset]);
    if (path === "/datasets/dataset-a/versions") return respond([version]);
    if (path.endsWith("/profile"))
      return respond({
        rows: 512,
        columns: [{ column: "demand", dtype: "float64", missing: 2 }],
        numeric_columns: ["demand", "price"],
        missing_cells: 2,
        duplicate_rows: 0,
        memory_bytes: 8192,
        statistics: [{ column: "demand", count: 510, mean: 100 }],
        correlation_columns: ["demand", "price"],
        correlations: [{ column: "demand", demand: 1, price: 0.5 }],
      });
    if (path.endsWith("/plot"))
      return respond({
        numeric: url.searchParams.get("column") !== "date",
        total: 512,
        sampled: false,
        points: [
          { x: 0, y: 100 },
          { x: 1, y: null },
          { x: 2, y: 120 },
        ],
        histogram: [
          { lower: 100, upper: 110, count: 300 },
          { lower: 110, upper: 120, count: 210 },
        ],
        categories: [{ value: "100", count: 300 }],
        non_null: 510,
      });
    if (path.endsWith("/preview") && path.startsWith("/datasets"))
      return respond({
        columns: ["date", "demand", "price"],
        rows: [{ date: "2026-09-01", demand: 100, price: 5 }],
        total: 512,
      });
    if (path === "/preview")
      return respond({
        quality: [{ dataset: "Retail demand", status: "ready", missing: 0 }],
        series: [
          {
            dataset: "Retail demand",
            rows: 512,
            columns: version.payload.columns,
          },
        ],
        columns: version.payload.columns,
        scenario_template: [
          {
            dataset: "Retail demand",
            row: 512,
            timestamp: "2026-10-01",
            covariate: "price",
            value: 5,
          },
        ],
      });
    if (path === "/drafts" && request.method() === "GET")
      return respond([draft]);
    if (path === "/drafts" && request.method() === "POST") {
      const body = request.postDataJSON();
      return respond(
        { ...draft, id: "draft-copy", payload: body.payload, name: body.name },
        201,
      );
    }
    if (path === "/drafts/draft-a" && request.method() === "PATCH") {
      const body = request.postDataJSON();
      if (body.revision !== draft.revision)
        return respond({ detail: "Draft changed in another tab." }, 409);
      draft = { ...draft, revision: draft.revision + 1, payload: body.payload };
      return respond(draft);
    }
    if (path === "/drafts/draft-a") return respond(draft);
    if (path === "/drafts/draft-copy")
      return respond({ ...draft, id: "draft-copy" });
    if (path === "/runs") return respond([run]);
    if (path === "/runs/run-a") return respond(run);
    if (path.endsWith("/summary"))
      return respond({
        metrics: { mae: 2, rmse: 3, smape_percent: 4, observations: 32 },
        scope: "holdout",
        period_start: "2026-10-01",
        period_end: "2026-10-28",
        delta: null,
      });
    if (path.endsWith("/input-context"))
      return respond({
        windows: [
          {
            dataset: "Retail demand",
            variant: "",
            origin: null,
            context_length: 1,
            horizon: 32,
            context_shape: [1, 1],
            past_only_shape: null,
            past_future_shape: [1, 33],
            sampled: false,
            lineage: [],
            signals: [
              { signal: "demand", role: "target", missing: 0 },
              { signal: "price", role: "known_future", missing: 0 },
            ],
          },
        ],
        rows: [
          {
            dataset: "Retail demand",
            signal: "price",
            role: "known_future",
            phase: "history",
            position: 0,
            timestamp: "2026-09-30",
            value: 5,
          },
          {
            dataset: "Retail demand",
            signal: "price",
            role: "known_future",
            phase: "future",
            position: 1,
            timestamp: "2026-10-01",
            value: 6,
          },
        ],
        events: [
          {
            signal: "calendar_event:promotion",
            start: "2026-10-03",
            end: "2026-10-04",
          },
        ],
      });
    if (path.endsWith("/replay"))
      return route.fulfill({
        contentType: "text/x-python",
        body: "# Creates a NEW job\nprint('replay')",
      });
    if (path === "/runs/run-a/chart")
      return respond({
        history: [{ timestamp: "2026-09-30", value: -22 }],
        forecast: forecastRows.slice(0, 32),
        datasets: ["Retail demand"],
        targets: ["demand"],
        references: ["last_value"],
        reference_rows: url.searchParams.get("reference")
          ? forecastRows.slice(0, 32).map((row) => ({ ...row, point: -22 }))
          : [],
      });
    if (path.startsWith("/runs/run-a/tables/")) {
      const offset = Number(url.searchParams.get("offset") ?? 0);
      return respond({
        columns: Object.keys(forecastRows[0]),
        rows: forecastRows.slice(offset, offset + 100),
        total: forecastRows.length,
      });
    }
    if (path === "/jobs" && request.method() === "POST") {
      submissions.push({
        body: request.postDataJSON(),
        key: request.headers()["idempotency-key"],
      });
      return respond(
        {
          id: "job-a",
          workspace_id: "local",
          kind: "forecast",
          status: "queued",
          stage: "Queued",
          spec: request.postDataJSON().spec,
          attempt: 1,
          result_id: null,
          error: null,
          created_at: timestamp,
          updated_at: timestamp,
        },
        202,
      );
    }
    if (path === "/jobs/job-a")
      return respond({
        id: "job-a",
        workspace_id: "local",
        kind: "forecast",
        status: "queued",
        stage: "Queued",
        spec,
        attempt: 1,
        result_id: null,
        error: null,
        created_at: timestamp,
        updated_at: timestamp,
      });
    if (["/jobs", "/tracking", "/models", "/workers"].includes(path))
      return respond([]);
    return respond({ detail: `Unmocked request ${path}` }, 404);
  });
  return { submissions, requests, currentDraft: () => draft };
}

test("dataset explorer pages rows and renders EDA charts on desktop and mobile", async ({
  page,
  context,
}) => {
  const mock = await fixture(context);
  await page.goto("/data?workspace=local&versions=version-a");
  await page.getByRole("button", { name: "Tail", exact: true }).click();
  await expect
    .poll(() => mock.requests.some((path) => path.includes("offset=502")))
    .toBe(true);
  await page.getByRole("button", { name: "Head", exact: true }).click();
  await expect(
    page.getByText("Rows 1–10 of 512.", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Statistics", exact: true }).click();
  await expect(
    page.getByRole("table", { name: "Descriptive statistics" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Charts", exact: true }).click();
  await page.getByLabel("Column / Y axis").selectOption("demand");
  for (const kind of ["Line", "Histogram", "Scatter", "Category counts"]) {
    await page.getByLabel("Chart type").selectOption(kind);
    await expect(
      page
        .getByRole("img", { name: `${kind} chart of demand.`, exact: false })
        .locator("canvas"),
    ).toBeVisible();
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await expect(
    page.getByRole("table", { name: "Column types and data quality" }),
  ).toBeVisible();
});

test("results expose reference zoom report and saved input signals without submitting jobs", async ({
  page,
  context,
}, testInfo) => {
  const mock = await fixture(context);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/forecasts?versions=version-a&draft=draft-a&run=run-a");
  await page.getByLabel("Reference curve").selectOption("last_value");
  await expect
    .poll(() =>
      mock.requests.some((path) => path.includes("reference=last_value")),
    )
    .toBe(true);
  await page
    .getByRole("button", { name: "Forecast horizon", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Forecast horizon", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByText("Holdout evaluation.", { exact: false }),
  ).toBeVisible();
  await expect(
    page.getByText("Known future input", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Reference curve").scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath("results-desktop.png") });
  await page
    .getByRole("button", { name: "Execution report", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toContainText("Recorded runtime");
  await page
    .getByRole("dialog")
    .getByText("Show the call", { exact: true })
    .click();
  await expect(page.getByRole("dialog")).toContainText("Creates a NEW job");
  await page.getByRole("button", { name: "Close details" }).click();
  await page.getByLabel("Color theme").selectOption("dark");
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  expect(mock.submissions).toHaveLength(0);
  expect(errors).toEqual([]);
  await page.getByLabel("Reference curve").scrollIntoViewIfNeeded();
  await page.screenshot({ path: testInfo.outputPath("results-mobile.png") });
});

test("signal changes invalidate preview and keep forecasting explicit", async ({
  page,
  context,
}) => {
  const mock = await fixture(context);
  await page.goto("/forecasts?versions=version-a&draft=draft-a");
  await page
    .getByRole("button", { name: "Preview data quality", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Run forecast", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("checkbox", { name: "price Known future", exact: true })
    .uncheck();
  await expect(
    page.getByRole("button", { name: "Run forecast", exact: true }),
  ).toBeDisabled();
  expect(mock.submissions).toHaveLength(0);
  await page
    .getByRole("button", { name: "Preview data quality", exact: true })
    .click();
  await page.getByRole("button", { name: "Run forecast", exact: true }).click();
  await expect.poll(() => mock.submissions.length).toBe(1);
  expect(
    (mock.submissions[0].body as { spec: { disabled_covariates: string[] } })
      .spec.disabled_covariates,
  ).toEqual(["price"]);
});

test("overview shows active tracking and saved accuracy from the latest evaluated run", async ({
  page,
  context,
}) => {
  await fixture(context);
  const evaluated = {
    ...run,
    id: "evaluated-a",
    name: "Actuals assessment",
    payload: { ...run.payload, kind: "assessment" },
  };
  await context.route("**/api/v1/runs?**", (route) =>
    route.fulfill({ json: [run, evaluated] }),
  );
  await context.route("**/api/v1/tracking?**", (route) =>
    route.fulfill({
      json: [
        record("tracking-a", "tracking", "Tracked demand", {
          run_id: "run-a",
          auto_refresh: false,
        }),
      ],
    }),
  );
  await context.route("**/api/v1/runs/evaluated-a/tables/metrics?**", (route) =>
    route.fulfill({
      json: {
        columns: [
          "dataset",
          "target",
          "observations",
          "mae",
          "rmse",
          "smape_percent",
        ],
        rows: [
          {
            dataset: "Retail demand",
            target: "demand",
            observations: 8,
            mae: 1.7146,
            rmse: 2.092,
            smape_percent: 2.676,
          },
        ],
        total: 1,
      },
    }),
  );
  await page.goto("/overview");
  await expect(
    page.getByText("1 tracked forecasts", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Latest saved accuracy metrics" }),
  ).toContainText("1.7146");
  await expect(
    page.getByRole("link", { name: "View evaluated run" }),
  ).toHaveAttribute("href", /run=evaluated-a/);
});

test("seven routes preserve selected analytical context and remain usable on narrow screens", async ({
  page,
  context,
}) => {
  await fixture(context);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/overview?versions=version-a&target=demand");
  for (const name of [
    "Data",
    "Forecasts",
    "Experiments",
    "Scenarios",
    "Tracking",
    "Models",
    "Overview",
  ]) {
    await page
      .getByRole("navigation", { name: "Main navigation" })
      .getByRole("link", { name, exact: true })
      .click();
    await expect(
      page.getByRole("heading", {
        level: 1,
        name: name === "Data" ? "Data library" : name,
      }),
    ).toBeVisible();
    expect(new URL(page.url()).searchParams.get("versions")).toBe("version-a");
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open navigation" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("button", { name: "Close details" }).click();
  expect(errors).toEqual([]);
});

test("the product shell remains readable from phone to wide desktop", async ({
  page,
  context,
}) => {
  await fixture(context);
  for (const width of [320, 390, 768, 1280, 1536]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/overview?versions=version-a");
    await expect(
      page.getByRole("heading", { level: 1, name: "Overview" }),
    ).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
  }

  await page.setViewportSize({ width: 320, height: 900 });
  await page.goto("/forecasts?run=run-a&draft=draft-a&versions=version-a");
  const horizon = page.getByLabel("Horizon", { exact: false });
  await expect(horizon).toBeVisible();
  expect(
    await horizon.evaluate((element) =>
      Number.parseFloat(getComputedStyle(element).fontSize),
    ),
  ).toBeGreaterThanOrEqual(16);
  await expect(
    page.getByText("Scroll sideways to view all columns"),
  ).toBeVisible();
  expect(
    await page
      .locator(".configuration-actions")
      .evaluate((element) => getComputedStyle(element).position),
  ).toBe("sticky");
  await page.evaluate(() =>
    document.documentElement.setAttribute("dir", "rtl"),
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
});

test("a stale tab cannot overwrite a saved draft and can recover a separate copy", async ({
  page,
  context,
}) => {
  const mock = await fixture(context);
  const second = await context.newPage();
  await page.goto("/forecasts?draft=draft-a&versions=version-a");
  await second.goto("/forecasts?draft=draft-a&versions=version-a");
  await expect(page.getByLabel("Horizon", { exact: false })).toHaveValue("32");
  await expect(second.getByLabel("Horizon", { exact: false })).toHaveValue(
    "32",
  );
  await page.getByLabel("Horizon", { exact: false }).fill("24");
  await expect.poll(() => mock.currentDraft().revision).toBe(2);
  await second.getByLabel("Horizon", { exact: false }).fill("48");
  await expect(
    second.getByText("Another tab saved this draft.", { exact: false }),
  ).toBeVisible();
  expect(mock.currentDraft().payload.spec.settings.horizon).toBe(24);
  await expect(second.getByLabel("Horizon", { exact: false })).toHaveValue(
    "48",
  );
  await second.getByRole("button", { name: "Save as a new draft" }).click();
  await expect(second).toHaveURL(/draft=draft-copy/);
  await second.close();
});

test("preview gates submission and a job captures the configuration before later edits", async ({
  page,
  context,
}) => {
  const mock = await fixture(context);
  await page.goto("/forecasts?draft=draft-a&versions=version-a");
  await expect(
    page.getByRole("button", { name: "Run forecast", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Preview data quality" }).click();
  await expect(
    page.getByRole("button", { name: "Run forecast", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Run forecast", exact: true }).click();
  await expect.poll(() => mock.submissions.length).toBe(1);
  await page.getByLabel("Horizon", { exact: false }).fill("64");
  expect(
    (mock.submissions[0].body as { spec: typeof spec }).spec.settings.horizon,
  ).toBe(32);
  expect(mock.submissions[0].key).toBeTruthy();
  await expect(
    page.getByRole("button", { name: "Run forecast", exact: true }),
  ).toBeDisabled();
});

test("result tables page on the server and chart renders without global page overflow", async ({
  page,
  context,
}) => {
  const mock = await fixture(context);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/forecasts?run=run-a&draft=draft-a&versions=version-a");
  await expect(page.getByRole("img", { name: /^demand:/ })).toBeVisible();
  await expect(page.getByText("1–100 of 125 rows")).toBeVisible();
  await page.getByRole("button", { name: "Next page", exact: true }).click();
  await expect(page.getByText("101–125 of 125 rows")).toBeVisible();
  expect(
    mock.requests.some(
      (path) => path.includes("offset=100") && path.includes("limit=100"),
    ),
  ).toBe(true);
  expect(mock.submissions).toHaveLength(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: "test-results/forecast-desktop.png",
    fullPage: true,
  });
  await page.getByLabel("Color theme").selectOption("dark");
  await page.screenshot({
    path: "test-results/forecast-dark.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("scenario edits remain tied to future cells in the submitted specification", async ({
  page,
  context,
}) => {
  const mock = await fixture(context);
  await page.goto("/scenarios?draft=draft-a&versions=version-a");
  await page.getByRole("button", { name: "Preview data quality" }).click();
  await page.getByRole("button", { name: "Add scenario", exact: true }).click();
  await page.getByLabel("Scenario 1, price, row 512").fill("7.5");
  await page.getByRole("button", { name: "Preview data quality" }).click();
  await page
    .getByRole("button", { name: "Run scenarios", exact: true })
    .click();
  await expect.poll(() => mock.submissions.length).toBe(1);
  expect(mock.submissions[0].body).toMatchObject({
    kind: "scenario",
    spec: {
      scenarios: [
        {
          name: "Scenario 1",
          overrides: [
            {
              dataset: "Retail demand",
              row: 512,
              covariate: "price",
              value: 7.5,
            },
          ],
        },
      ],
    },
  });
});

test("configuration comparisons keep their experiment selection across navigation", async ({
  page,
  context,
}) => {
  const mock = await fixture(context);
  await page.goto("/experiments?draft=draft-a&versions=version-a");
  await page.getByLabel("Experiment type").selectOption("settings");
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await page.getByText("Configuration 2", { exact: true }).click();
  await page
    .getByRole("spinbutton", { name: "Context Historical rows", exact: true })
    .last()
    .fill("128");
  await page.getByRole("button", { name: "Preview data quality" }).click();
  await page
    .getByRole("button", { name: "Run experiment", exact: true })
    .click();
  await expect.poll(() => mock.submissions.length).toBe(1);
  const submitted = mock.submissions[0].body as {
    kind: string;
    spec: typeof spec;
  };
  expect(submitted.kind).toBe("settings");
  expect(
    submitted.spec.configurations.map((item) => item.settings.context_length),
  ).toEqual([512, 128]);
  await page
    .getByRole("navigation", { name: "Main navigation" })
    .getByRole("link", { name: "Overview", exact: true })
    .click();
  await page
    .getByRole("navigation", { name: "Main navigation" })
    .getByRole("link", { name: "Experiments", exact: true })
    .click();
  await expect(page.getByLabel("Experiment type")).toHaveValue("settings");
});

test("structured worker failures are readable and retry is an explicit action", async ({
  page,
  context,
}) => {
  await fixture(context);
  let retried = false;
  const failed = {
    id: "failed-job",
    workspace_id: "local",
    kind: "forecast",
    status: "failed",
    stage: "inference",
    spec,
    attempt: 1,
    result_id: null,
    error: {
      code: "out_of_memory",
      message: "GPU memory is exhausted. Reduce context or batch size.",
    },
    created_at: timestamp,
    updated_at: timestamp,
  };
  await context.route("**/api/v1/jobs?**", (route) =>
    route.fulfill({ json: [failed] }),
  );
  await context.route("**/api/v1/jobs/failed-job/retry", (route) => {
    retried = true;
    return route.fulfill({
      json: { ...failed, status: "queued", attempt: 2, error: null },
    });
  });
  await page.goto("/overview");
  await expect(page.getByText(failed.error.message)).toBeVisible();
  expect(retried).toBe(false);
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await expect.poll(() => retried).toBe(true);
});

test("retrying an ambiguous submission reuses its idempotency key", async ({
  page,
  context,
}) => {
  await fixture(context);
  const keys: string[] = [];
  await context.route("**/api/v1/jobs", async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    keys.push(route.request().headers()["idempotency-key"]);
    if (keys.length === 1)
      return route.fulfill({
        status: 502,
        json: {
          detail:
            "The submission response was interrupted. Retry to resolve the same job.",
        },
      });
    return route.fallback();
  });
  await page.goto("/forecasts?draft=draft-a&versions=version-a");
  await page.getByRole("button", { name: "Preview data quality" }).click();
  await page.getByRole("button", { name: "Run forecast", exact: true }).click();
  await expect(
    page.getByText("The submission response was interrupted.", {
      exact: false,
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Run forecast", exact: true }).click();
  await expect.poll(() => keys.length).toBe(2);
  expect(keys[0]).toBeTruthy();
  expect(keys[1]).toBe(keys[0]);
});

test("a one-step forecast retains visible point and interval geometry", async ({
  page,
  context,
}) => {
  await fixture(context);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await context.route("**/api/v1/runs/run-a/chart**", (route) =>
    route.fulfill({
      json: {
        history: [
          { timestamp: "2026-09-29", value: -21 },
          { timestamp: "2026-09-30", value: -22 },
        ],
        forecast: forecastRows.slice(0, 1),
        targets: ["demand"],
        datasets: ["Retail demand"],
      },
    }),
  );
  await page.goto("/forecasts?run=run-a&draft=draft-a&versions=version-a");
  await expect(
    page.getByRole("img", { name: /1 forecast steps/ }),
  ).toBeVisible();
  await page
    .getByRole("figure")
    .screenshot({ path: "test-results/forecast-one-step.png" });
  expect(errors).toEqual([]);
});

test("automatic refresh is off by default and updating it preserves the issued run and revision", async ({
  page,
  context,
}) => {
  await fixture(context);
  let tracked = record("tracking-a", "tracking", "Tracked demand", {
    run_id: "run-a",
    auto_refresh: false,
    associations: {},
  });
  const patches: { revision: number; payload: typeof tracked.payload }[] = [];
  await context.route("**/api/v1/tracking?**", (route) =>
    route.fulfill({ json: [tracked] }),
  );
  await context.route("**/api/v1/runs/run-a", (route) =>
    route.fulfill({
      json: {
        ...run,
        payload: {
          ...run.payload,
          manifest: { datasets: [{ dataset_id: "Retail demand" }] },
        },
      },
    }),
  );
  await context.route("**/api/v1/tracking/tracking-a", (route) => {
    const body = route.request().postDataJSON();
    patches.push(body);
    tracked = {
      ...tracked,
      payload: body.payload,
      revision: tracked.revision + 1,
    };
    return route.fulfill({ json: tracked });
  });
  await page.goto("/tracking?versions=version-a");
  await expect(
    page.getByRole("checkbox", { name: /^Automatically refresh forecasts/ }),
  ).not.toBeChecked();
  await page.getByRole("button", { name: /Tracked demand/ }).click();
  const checkbox = page.getByRole("checkbox", {
    name: /^Automatically refresh this forecast/,
  });
  await expect(checkbox).not.toBeChecked();
  await checkbox.check();
  await expect(checkbox).toBeChecked();
  expect(patches[0]).toMatchObject({
    revision: 1,
    payload: { run_id: "run-a", auto_refresh: true },
  });
  await checkbox.uncheck();
  await expect(checkbox).not.toBeChecked();
  expect(patches[1]).toMatchObject({
    revision: 2,
    payload: { run_id: "run-a", auto_refresh: false },
  });
});

test("calibration reads persisted coverage at the selected horizon", async ({
  page,
  context,
}) => {
  await fixture(context);
  const steps: string[] = [];
  const rows = [20, 40, 60, 80].map((nominal) => ({
    nominal_coverage_percent: nominal,
    observed_coverage_percent: nominal - 5,
    observations: 20,
    mean_width: 3,
    missing_actuals: 0,
    invalid_bounds: 0,
    crossings: 0,
  }));
  await context.route("**/api/v1/runs/run-a/tables/calibration**", (route) => {
    steps.push(new URL(route.request().url()).searchParams.get("step") ?? "");
    return route.fulfill({
      json: { columns: Object.keys(rows[0]), rows, total: 4 },
    });
  });
  await page.goto("/forecasts?run=run-a&draft=draft-a&versions=version-a");
  await page.getByRole("button", { name: "Inspect coverage" }).click();
  await expect(
    page.getByRole("img", {
      name: /Observed coverage against nominal interval coverage/,
    }),
  ).toBeVisible();
  await page.getByRole("spinbutton", { name: "Horizon step" }).fill("4");
  await expect.poll(() => steps.includes("4")).toBe(true);
  await expect(
    page.getByRole("region", { name: "Saved interval calibration values" }),
  ).toBeVisible();
  await page
    .getByRole("img", {
      name: /Observed coverage against nominal interval coverage/,
    })
    .screenshot({ path: "test-results/calibration-coverage.png" });
});
