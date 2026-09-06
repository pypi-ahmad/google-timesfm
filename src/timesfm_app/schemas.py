"""HTTP contracts independent of Streamlit and heavyweight model construction."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
  model_config = ConfigDict(extra="forbid")


class Mapping(Contract):
  timestamp: str | None = None
  targets: list[str] = Field(default_factory=list)
  past_only: list[str] = Field(default_factory=list)
  past_future: list[str] = Field(default_factory=list)

  @model_validator(mode="after")
  def disjoint_roles(self):
    columns = self.targets + self.past_only + self.past_future
    if len(set(columns)) != len(columns) or self.timestamp in columns:
      raise ValueError("Each column must have one role.")
    return self


class ForecastSettings(Contract):
  horizon: int = Field(default=32, ge=1, le=15360)
  context_length: int = Field(default=512, ge=1, le=15360)
  task: Literal["forecast", "holdout"] = "forecast"
  mode: Literal["multivariate", "univariate"] = "multivariate"
  return_quantiles: bool = True
  use_symmetric_averaging: bool = True
  make_positive: bool = True
  sort_quantiles: bool = True
  use_znorm: bool = False
  padding_mode: Literal["none", "edge"] = "none"
  batch_size: int = Field(default=4, ge=1, le=64)
  allow_benchmark_chunking: bool = False


class CalendarEvent(Contract):
  name: str
  start: str
  end: str | None = None


class Calendar(Contract):
  weekday: bool = False
  month: bool = False
  holiday_country: str | None = None
  holiday_subdivision: str | None = None
  events: list[CalendarEvent] = Field(default_factory=list)


class Preparation(Contract):
  group_columns: list[str] = Field(default_factory=list)
  frequency: str | None = None
  calendar: Calendar = Field(default_factory=Calendar)
  excluded_groups: list[str] = Field(default_factory=list)


class ModelSelection(Contract):
  kind: Literal["hub", "local"] = "hub"
  source: str = "google/timesfm-3.0-pytorch"
  revision: str | None = None
  offline: bool = False


class Analysis(Contract):
  windows: int = Field(default=5, ge=1, le=1000)
  stride: int | None = Field(default=None, ge=1)
  seasonal_period: int = Field(default=1, ge=1, le=15360)
  selected_covariates: list[str] = Field(default_factory=list)


class Configuration(Contract):
  name: str = Field(min_length=1)
  settings: ForecastSettings


class ScenarioEdit(Contract):
  dataset: str
  row: int = Field(ge=0)
  covariate: str
  value: float = Field(allow_inf_nan=False)


class Scenario(Contract):
  name: str = Field(min_length=1)
  overrides: list[ScenarioEdit] = Field(default_factory=list)


JobKind = Literal[
  "forecast",
  "backtest",
  "joint_independent",
  "covariates",
  "settings",
  "baselines",
  "scenario",
  "anomaly",
  "assessment",
  "model_check",
]


class RunSpec(Contract):
  kind: JobKind = "forecast"
  dataset_version_ids: list[str] = Field(default_factory=list)
  mapping: Mapping = Field(default_factory=Mapping)
  settings: ForecastSettings = Field(default_factory=ForecastSettings)
  preparation: Preparation = Field(default_factory=Preparation)
  model: ModelSelection = Field(default_factory=ModelSelection)
  analysis: Analysis = Field(default_factory=Analysis)
  configurations: list[Configuration] = Field(default_factory=list, max_length=8)
  scenarios: list[Scenario] = Field(default_factory=list, max_length=3)
  parent_run_id: str | None = None
  associations: dict[str, str] = Field(default_factory=dict)

  @model_validator(mode="after")
  def validate_roles(self):
    roles = self.mapping.targets + self.mapping.past_only + self.mapping.past_future
    if set(roles).intersection(self.preparation.group_columns):
      raise ValueError("Group identifiers cannot also be targets or covariates.")
    if len(set(self.dataset_version_ids)) != len(self.dataset_version_ids):
      raise ValueError("Dataset versions must be distinct.")
    return self


class JobSubmission(Contract):
  workspace_id: str = "local"
  kind: JobKind = "forecast"
  spec: RunSpec


class RecordCreate(Contract):
  workspace_id: str = "local"
  name: str = Field(default="Untitled", min_length=1, max_length=200)
  payload: dict[str, Any] = Field(default_factory=dict)


class RecordPatch(Contract):
  name: str | None = Field(default=None, min_length=1, max_length=200)
  payload: dict[str, Any]
  revision: int = Field(ge=1)


class ActualsSubmission(Contract):
  dataset_version_ids: list[str] = Field(min_length=1)
  associations: dict[str, str] = Field(default_factory=dict)


class TrackingCreate(RecordCreate):
  pass


class RetentionPolicy(Contract):
  enabled: bool = False
  max_age_days: int | None = Field(default=None, ge=1, le=36500)
  max_runs: int | None = Field(default=None, ge=1, le=1000000)


class RecordResponse(Contract):
  id: str
  workspace_id: str
  kind: str
  name: str
  revision: int
  payload: dict[str, Any]
  created_at: str


class JobResponse(Contract):
  id: str
  workspace_id: str
  kind: JobKind
  status: Literal["queued", "running", "cancelling", "succeeded", "failed", "cancelled"]
  stage: str
  spec: dict[str, Any]
  idempotency_key: str
  request_hash: str
  attempt: int
  lease_until: str | None
  cancel_requested: bool
  result_id: str | None
  error: dict[str, Any] | None
  created_at: str
  updated_at: str


class TablePage(Contract):
  columns: list[str]
  rows: list[dict[str, Any]]
  total: int
