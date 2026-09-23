# Graph Report - google-timesfm  (2026-09-23)

## Corpus Check
- 265 files · ~420,863 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 20 file(s) not represented in the graph (top: .csv 5, (none) 4, .ipynb 4)

## Summary
- 2648 nodes · 6083 edges · 164 communities (129 shown, 35 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 388 edges (avg confidence: 0.92)
- Token cost: 865 input · 13,425 output

## Community Hubs (Navigation)
- TimesFM-3 Run Store
- TimesFM-3 Model Loading
- Workbench Store
- Python Dependencies Pandas
- Python Dependencies Numpy
- Workbench Artifacts
- TimesFM-3 Explorer
- Workbench Api
- TimesFM-3 Timesfm3 Forecaster
- Tests App Jobs
- Workbench UI Components
- TimesFM-3 Analysis Ui
- Tests Analysis
- TimesFM Timesfm 2P5 Flax
- Workbench UI Lib
- Forecasting Examples Check System
- Tests Explorer
- TimesFM-3 Data Preparation
- Workbench Schemas
- Workbench UI Tanstack React Query
- Workbench UI Lib
- Workbench Worker
- Workbench Native
- Workbench Migration
- Python Dependencies Tensor
- Workbench Services
- Python Dependencies Tensor
- Python Dependencies Os
- Workbench UI Lib
- Python Dependencies Math
- Workbench Store
- Workbench UI React
- Workbench Config
- TimesFM-3 Explorer
- Workbench Artifacts
- TimesFM Timesfm 2P5 Torch
- Forecasting Examples Finetuning
- Tests App Native
- Workbench Schemas
- Tests Analysis Extensions
- Tests App Services
- Forecasting Examples Covariates Forecasting
- TimesFM Xreg Lib
- Workbench UI Dependencies
- Tests App Jobs
- TimesFM-3 Timesfm3 Forecaster
- Python Dependencies Tensor
- TimesFM Normalization
- Tests Base Utils
- Tests Diagnostic Client
- Documentation Diagrams
- TimesFM-3 Model
- TimesFM Util
- Workbench UI Tsconfig
- Documentation Timesfm3 Guide
- Python Dependencies Pathlib
- Research Knowledge Times Fm Source
- TimesFM Configs
- TimesFM Timesfm 2P5 Base
- TimesFM Util
- Python Dependencies Argparse
- TimesFM Timesfm 2P5 Flax
- TimesFM Configs
- Workbench UI Components
- Documentation Diagrams
- Research Knowledge Project Knowledge Index
- Forecasting Examples Forecast Csv
- TimesFM-3 Explorer
- Documentation Explanation
- Workbench Api
- Tests Forecast Csv
- TimesFM-3 Primitives
- TimesFM-3 Transformer
- Documentation Diagrams
- TimesFM Transformer
- Workbench Dataset Explorer
- Tests App Services
- TimesFM Timesfm 2P5 Base
- TimesFM Timesfm 2P5 Torch
- Python Dependencies Matplotlib Pyplot
- Forecasting Examples Anomaly Detection
- TimesFM Util
- Forecasting Examples Anomaly Detection
- Python Dependencies Uuid
- Documentation Diagrams
- Documentation Diagrams
- Workbench Config
- TimesFM Dense
- Tests Torch Layers
- Forecasting Examples Data Preparation Md
- TimesFM Util
- Tests Configs
- TimesFM Configs
- TimesFM Normalization
- Python Dependencies Tensor
- Forecasting Examples Covariates Forecasting
- Forecasting Examples Global Temperature
- Workbench UI Dev Dependencies
- Documentation Codebase
- Project Review
- TimesFM Dense
- Tests App Services
- Forecasting Examples Global Temperature
- Workbench UI Readme
- Documentation Readme
- Documentation Diagrams
- TimesFM-3 Dense
- TimesFM Timesfm 2P5 Flax
- TimesFM-3 Benchmarks Fev Bench
- Workbench UI Scripts
- Workbench UI Workspace Spec
- Workbench Worker
- TimesFM Normalization
- TimesFM-3 Model
- TimesFM-3 Normalization
- TimesFM Transformer
- Forecasting Examples Api Reference Md
- Workbench UI Lib
- Repository Automation Copilot Instructions
- Project Readme
- Research Knowledge Timesfm3 Api And
- Tests Base Utils
- Documentation Diagrams
- TimesFM-3 Cpm Revin Refine
- TimesFM Timesfm 2P5 Torch
- Tests App Services
- Workbench UI Next Env D
- Documentation Diagrams
- Documentation Diagrams
- Project Prometheus
- Project Requirements
- TimesFM-3 Tracking
- Tests App Native
- Tests App Services
- Forecasting Examples Global Temperature
- Documentation Diagrams
- Documentation How To
- Repository Automation Main
- Repository Automation Manual Publish
- Repository Automation Workbench
- Workbench Local Durable Application Services
- Tests App Native
- Tests App Native
- Forecasting Examples Finetuning
- Forecasting Examples Global Temperature
- Project Contributing
- Documentation Diagrams
- Documentation Diagrams
- Documentation Diagrams
- Documentation Diagrams
- Documentation Diagrams
- Documentation Diagrams
- Documentation Diagrams
- Documentation Diagrams
- Documentation Diagrams
- Documentation Diagrams
- Documentation Diagrams
- Documentation Diagrams
- Documentation How To
- Project Otel Collector
- Project Pyproject

## God Nodes (most connected - your core abstractions)
1. `create_app()` - 95 edges
2. `Store` - 84 edges
3. `ExplorerError` - 72 edges
4. `UploadedDataset` - 38 edges
5. `api()` - 38 edges
6. `DatasetMapping` - 35 edges
7. `execute_spec()` - 35 edges
8. `ArtifactStore` - 32 edges
9. `Conflict` - 32 edges
10. `RunStoreError` - 31 edges

## Surprising Connections (you probably didn't know these)
- `TimesFM-3 Python API and Input Shapes` --semantically_similar_to--> `How to Use the TimesFM-3 Python API`  [INFERRED] [semantically similar]
  knowledge/concepts/timesfm3-api-and-usage.md → docs/how-to/use-python-api.md
- `RunArtifact` --uses--> `_render_comparison()`  [INFERRED]
  src/timesfm3/explorer.py → streamlit_app.py
- `RunStoreError` --uses--> `test_app_warns_when_persistence_is_unavailable()`  [INFERRED]
  src/timesfm3/run_store.py → tests/test_streamlit_app.py
- `GPUProcessLock` --uses--> `test_gpu_lock_excludes_second_process_without_loading_torch()`  [INFERRED]
  src/timesfm_app/worker.py → tests/test_app_services.py
- `Store` --uses--> `test_production_store_rejects_sqlite()`  [INFERRED]
  src/timesfm_app/store.py → tests/test_app_store.py

## Import Cycles
- 3-file cycle: `src/timesfm/__init__.py -> src/timesfm/timesfm_2p5/timesfm_2p5_flax.py -> src/timesfm/flax/dense.py -> src/timesfm/__init__.py`
- 3-file cycle: `src/timesfm/__init__.py -> src/timesfm/timesfm_2p5/timesfm_2p5_flax.py -> src/timesfm/flax/transformer.py -> src/timesfm/__init__.py`
- 3-file cycle: `src/timesfm/__init__.py -> src/timesfm/timesfm_2p5/timesfm_2p5_flax.py -> src/timesfm/timesfm_2p5/timesfm_2p5_base.py -> src/timesfm/__init__.py`
- 3-file cycle: `src/timesfm/__init__.py -> src/timesfm/timesfm_2p5/timesfm_2p5_torch.py -> src/timesfm/timesfm_2p5/timesfm_2p5_base.py -> src/timesfm/__init__.py`
- 3-file cycle: `src/timesfm/__init__.py -> src/timesfm/timesfm_2p5/timesfm_2p5_torch.py -> src/timesfm/torch/dense.py -> src/timesfm/__init__.py`
- 3-file cycle: `src/timesfm/__init__.py -> src/timesfm/timesfm_2p5/timesfm_2p5_torch.py -> src/timesfm/torch/transformer.py -> src/timesfm/__init__.py`
- 3-file cycle: `src/timesfm3/__init__.py -> src/timesfm3/evaluator.py -> src/timesfm3/timesfm3_forecaster.py -> src/timesfm3/__init__.py`
- 4-file cycle: `src/timesfm3/__init__.py -> src/timesfm3/evaluator.py -> src/timesfm3/timesfm3_forecaster.py -> src/timesfm3/model.py -> src/timesfm3/__init__.py`
- 5-file cycle: `src/timesfm3/__init__.py -> src/timesfm3/evaluator.py -> src/timesfm3/timesfm3_forecaster.py -> src/timesfm3/model.py -> src/timesfm3/cpm_revin_refine.py -> src/timesfm3/__init__.py`
- 5-file cycle: `src/timesfm3/__init__.py -> src/timesfm3/evaluator.py -> src/timesfm3/timesfm3_forecaster.py -> src/timesfm3/model.py -> src/timesfm3/transformer.py -> src/timesfm3/__init__.py`
- 5-file cycle: `src/timesfm3/__init__.py -> src/timesfm3/evaluator.py -> src/timesfm3/timesfm3_forecaster.py -> src/timesfm3/model.py -> src/timesfm3/dense.py -> src/timesfm3/__init__.py`

## Hyperedges (group relationships)
- **Legacy Explorer upload-to-artifact sequence** — docs_codebase_architecture_parse_upload, docs_codebase_architecture_prepare_batch, docs_codebase_architecture_resolve_model, docs_codebase_architecture_run_forecast, docs_codebase_architecture_make_run_artifact [EXTRACTED 1.00]
- **Local Explorer Process** — docs_diagrams_timesfm3_explorer_architecture_visual_check_1440x900_dark_streamlit_explorer, docs_diagrams_timesfm3_explorer_architecture_visual_check_1440x900_dark_preparation_and_validation, docs_diagrams_timesfm3_explorer_architecture_visual_check_1440x900_dark_resolved_timesfm_3_model, docs_diagrams_timesfm3_explorer_architecture_visual_check_1440x900_dark_results_and_exports, docs_diagrams_timesfm3_explorer_architecture_visual_check_1440x900_dark_historical_analysis, docs_diagrams_timesfm3_explorer_architecture_visual_check_1440x900_dark_local_duckdb, docs_how_to_timesfm3_explorer_handbook_forecast_tracking [EXTRACTED 1.00]
- **Components of the TimesFM-3 local workbench** — docs_diagrams_workbench_architecture_react_workbench, docs_diagrams_workbench_architecture_fastapi, docs_diagrams_workbench_architecture_visual_check_1440x900_dark_postgresql, docs_diagrams_workbench_architecture_outbox_dispatcher, docs_diagrams_workbench_architecture_visual_check_2048x1320_dark_memurai_dramatiq, docs_diagrams_workbench_architecture_inference_worker, docs_diagrams_workbench_architecture_timesfm_3_services, docs_codebase_integrations_artifact_store [EXTRACTED 1.00]
- **TimesFM-3 evaluation across public forecasting benchmarks** — knowledge_references_google_timesfm_3_blog_google_research_timesfm_3_launch_post, knowledge_references_fev_bench_leaderboard_fev_bench_leaderboard, knowledge_references_gift_eval_leaderboard_gift_eval_leaderboard, knowledge_references_time_leaderboard_time_leaderboard, knowledge_references_timesfm3_benchmark_assets_timesfm_3_benchmark_reproduction_assets [EXTRACTED 1.00]
- **TimesFM 3.0 evaluation across benchmark suites** — timesfm3_usage_benchmarks_readme_timesfm_3_0, timesfm3_usage_benchmarks_readme_autogluon_fev_bench, timesfm3_usage_benchmarks_readme_icml_time_benchmark, timesfm3_usage_benchmarks_readme_salesforce_gift_eval [EXTRACTED 1.00]
- **Local workbench asynchronous execution architecture** — docs_diagrams_workbench_architecture_fastapi, docs_diagrams_workbench_architecture_outbox_dispatcher, docs_diagrams_workbench_architecture_memurai_and_dramatiq_queues, docs_diagrams_workbench_architecture_inference_worker, docs_diagrams_workbench_architecture_timesfm_3_services [INFERRED 0.85]
- **TimesFM-3 Python API guidance** — docs_how_to_use_python_api_how_to_use_the_timesfm_3_python_api, docs_reference_python_api_timesfm_3_python_api_reference, knowledge_concepts_timesfm3_api_and_usage_timesfm_3_python_api_and_input_shapes [INFERRED 0.85]
- **Components of TimesFM-3 multivariate single-pass forecasting** — knowledge_references_google_timesfm_3_blog_timesfm_3_multivariate_architecture, knowledge_references_google_timesfm_3_blog_contiguous_patch_masking_forecasting, knowledge_references_huggingface_timesfm_3_model_card_official_timesfm_3_0_pytorch_checkpoint [INFERRED 0.85]
- **Legacy Explorer upload and derived-run persistence behavior** — planning_artifacts_reports_doc_sync_20260902_152911_duckdb_derived_run_retention, planning_artifacts_reports_doc_sync_20260902_152911_session_scoped_upload_data, docs_troubleshooting_duckdb_history_and_session_fallback, docs_codebase_architecture_duckdb_retention_and_session_only_inputs, docs_diagrams_timesfm3_explorer_architecture_explorer_duckdb_persistence [INFERRED 0.85]
- **Retained Streamlit Explorer documentation** — docs_explanation_architecture_architecture_and_data_flow, docs_how_to_timesfm3_explorer_handbook_legacy_timesfm_3_explorer_handbook, docs_how_to_use_streamlit_explorer_legacy_streamlit_explorer_guide, docs_reference_explorer_legacy_explorer_reference [INFERRED 0.95]
- **Native React and FastAPI workbench documentation** — docs_explanation_workbench_architecture_react_and_fastapi_workbench_architecture, docs_how_to_native_workbench_run_the_native_windows_workbench, docs_reference_workbench_api_workbench_api_reference, docs_tutorials_first_forecast_your_first_timesfm_3_workbench_forecast [INFERRED 0.95]

## Communities (164 total, 35 thin omitted)

### Community 0 - "TimesFM-3 Run Store"
Cohesion: 0.06
Nodes (112): dialog, duckdb, DuckDBPyConnection, re, RuntimeError, AnalysisArtifact, Derived results; no original uploads or historical context arrays. This…, artifact_zip() (+104 more)

### Community 1 - "TimesFM-3 Model Loading"
Cohesion: 0.05
Nodes (53): cache_resource, SimpleNamespace, load_forecaster(), Load official TimesFM-3 evaluator; caller owns resource caching., _digest(), load_resolved_model(), ModelSelection, Any (+45 more)

### Community 2 - "Workbench Store"
Cohesion: 0.12
Nodes (18): Session, _contains(), _dict(), _iso(), Job, _json_copy(), _now(), Any (+10 more)

### Community 3 - "Python Dependencies Pandas"
Cohesion: 0.06
Nodes (53): collections_abc, hashlib, io, numbers, pandas, analysis_metrics(), _baseline_table(), ForecastTask (+45 more)

### Community 4 - "Python Dependencies Numpy"
Cohesion: 0.07
Nodes (36): collections, d_ai_github_google_timesfm_src_timesfm_timesfm_2p5_py, dataclasses, huggingface_hub, numpy, pytest, safetensors_torch, Abstract configs for TimesFM-3 layers. Framework-agnostic dataclasses consumed… (+28 more)

### Community 5 - "Workbench Artifacts"
Cohesion: 0.07
Nodes (36): base64, pyarrow, pyarrow_parquet, ArtifactStore, _decode_cell(), _descriptor(), _encode_cell(), _frame_bytes() (+28 more)

### Community 6 - "TimesFM-3 Explorer"
Cohesion: 0.09
Nodes (47): date, importlib_metadata, _observed_end(), Freezes a validated, timestamp-normalized copy of each dataset. The returned…, Row index one past the last row with any observed target value. Exclusive bound…, _snapshots(), _calendar_columns(), _calendar_metadata() (+39 more)

### Community 7 - "Workbench Api"
Cohesion: 0.05
Nodes (22): demo_dataset(), Return a deterministic upload-shaped demo dataframe., create_app(), add_model(), assess(), check_model(), datasets(), demo() (+14 more)

### Community 8 - "TimesFM-3 Timesfm3 Forecaster"
Cohesion: 0.07
Nodes (35): ResidualBlockConfig, StackedTransformersConfig, TransformerConfig, ndarray, Evaluator subclass extending TimesFM3Forecaster for benchmark evaluation.…, Runs inference on a batch of time series with official benchmark defaults &…, TimesFM3Evaluator, TimesFM3 PyTorch API. Package entry point for the TimesFM v3 model line. Re-… (+27 more)

### Community 9 - "Tests App Jobs"
Cohesion: 0.08
Nodes (33): Event, logging, dispatch_loop(), dispatch_once(), main(), Deliver the PostgreSQL outbox to Dramatiq; duplicate delivery is safe. Polls…, Mark only acknowledged sends; a crash after send can safely deliver twice., create() (+25 more)

### Community 10 - "Workbench UI Components"
Cohesion: 0.15
Nodes (29): lucide-react, @radix-ui/react-dialog, react-hook-form, ConfigurationEditor(), Role, SettingsFields(), InputPreview(), JobStatus() (+21 more)

### Community 11 - "TimesFM-3 Analysis Ui"
Cohesion: 0.09
Nodes (40): analysis_fingerprint(), analysis_zip(), AnalysisSettings, anomaly_table(), comparison_table(), configuration_rankings(), ExperimentConfiguration, prepare_analysis() (+32 more)

### Community 12 - "Tests Analysis"
Cohesion: 0.10
Nodes (35): AnalysisKind, ScenarioEdit, dataset(), execute(), Any, DataFrame, parametrize, Behavioral checks for leakage-free TimesFM-3 analyses. (+27 more)

### Community 13 - "TimesFM Timesfm 2P5 Flax"
Cohesion: 0.09
Nodes (34): d_ai_github_google_timesfm_src_timesfm_timesfm_2p5_init_py, einshape, flax, flax_nnx_nn, functools, gc, jax, jax_numpy (+26 more)

### Community 14 - "Workbench UI Lib"
Cohesion: 0.06
Nodes (35): class-variance-authority, clsx, @hookform/resolvers, openapi-typescript, prettier, @radix-ui/react-slot, react-dom, tailwind-merge (+27 more)

### Community 15 - "Forecasting Examples Check System"
Cohesion: 0.09
Nodes (34): platform, shutil, struct, check_dataset_fit(), check_disk(), check_gpu(), check_package(), check_python() (+26 more)

### Community 16 - "Tests Explorer"
Cohesion: 0.15
Nodes (36): FakePredictor, mapping(), DataFrame, ForecastSettings, MonkeyPatch, parametrize, Path, Create an upload through the real CSV parser. (+28 more)

### Community 17 - "TimesFM-3 Data Preparation"
Cohesion: 0.17
Nodes (31): prepare_sources(), quality_report(), Prepare selected groups and calendar values without filling uploaded targets., Report readiness and warnings per group; never alter or interpolate data., daily(), DataFrame, MonkeyPatch, parametrize (+23 more)

### Community 18 - "Workbench Schemas"
Cohesion: 0.11
Nodes (31): BaseModel, contextlib, FastAPI, fastapi_encoders, fastapi_responses, sqlalchemy_exc, main(), Versioned local HTTP API. Inference runs exclusively in durable workers.… (+23 more)

### Community 19 - "Workbench UI Tanstack React Query"
Cohesion: 0.13
Nodes (19): ref_generated_api, next, @playwright/test, @tanstack/react-query, nextConfig, JobList(), ErrorNotice(), WorkerSummary() (+11 more)

### Community 20 - "Workbench UI Lib"
Cohesion: 0.14
Nodes (25): CalibrationContent(), CalibrationPanel(), DataTable(), features, RemoteTable(), DatasetChart, DatasetExplorer(), PlotData (+17 more)

### Community 21 - "Workbench Worker"
Cohesion: 0.09
Nodes (30): actor, ctypes, dramatiq, dramatiq_brokers_redis, dramatiq_middleware, prometheus_client, CancellationRequested, Exception (+22 more)

### Community 22 - "Workbench Native"
Cohesion: 0.14
Nodes (31): getpass, bootstrap(), cleanup_previous(), confirmed_gone(), creation_filetime(), doctor(), existing(), http_ready() (+23 more)

### Community 23 - "Workbench Migration"
Cohesion: 0.10
Nodes (24): concurrent_futures, import_id(), import_legacy(), _import_legacy(), persist(), save(), main(), Any (+16 more)

### Community 24 - "Python Dependencies Tensor"
Cohesion: 0.09
Nodes (27): safetensors, cpm_iterative_revin_refine(), Tensor, Standalone iterative RevIN refinement for CPM-masked patches in PyTorch.…, Refines RevIN stats at CPM-masked patches via iterative estimation. For each…, DecodeCache, get_activation_fn(), get_output_patch_via_roll() (+19 more)

### Community 25 - "Workbench Services"
Cohesion: 0.15
Nodes (26): One authored override at a sorted, zero-based uploaded row., ScenarioEdit, last_value_baseline(), Capture pre-interpolation inputs; never expose held-out targets as inputs., snapshot_batch(), _acquire_model(), _CheckedPredictor, _checkpoint() (+18 more)

### Community 26 - "Python Dependencies Tensor"
Cohesion: 0.10
Nodes (20): make_attn_mask(), make_segment_mask(), MixingTransformer, MultiHeadAttention, DecodeCache, StackedTransformersConfig, Tensor, TransformerConfig (+12 more)

### Community 27 - "Python Dependencies Os"
Cohesion: 0.08
Nodes (21): csv, gift_eval_data, gluonts_ev_metrics, gluonts_itertools, gluonts_model, gluonts_model_forecast, gluonts_time_feature, os (+13 more)

### Community 28 - "Workbench UI Lib"
Cohesion: 0.26
Nodes (23): ForecastFields(), JobActions(), RunViewer(), Shell(), SettingsContent(), DataPage(), ForecastsPage(), submissionKey() (+15 more)

### Community 29 - "Python Dependencies Math"
Cohesion: 0.08
Nodes (12): itertools, math, sklearn, Normalization layers for TimesFM3 PyTorch. Currently just `PerDimScale`, the…, NormalizationTest, Tests for TimesFM3 PyTorch primitives., Tests reversible transformations., ResidualBlockTest (+4 more)

### Community 30 - "Workbench Store"
Cohesion: 0.15
Nodes (23): LookupError, NotFound, QuotaExceeded, The requested record or job does not exist., The workspace already has its permitted number of active jobs., _associations(), Any, Reconcile tracked forecasts against new immutable logical dataset versions.… (+15 more)

### Community 31 - "Workbench UI React"
Cohesion: 0.14
Nodes (17): echarts, geist, next-themes, react, web_src_app_globals, metadata, viewport, CalibrationChart (+9 more)

### Community 32 - "Workbench Config"
Cohesion: 0.11
Nodes (16): BaseSettings, fastapi_testclient, pydantic, pydantic_settings, Native application configuration; credentials never enter browser settings.…, Settings, client(), fixture (+8 more)

### Community 33 - "TimesFM-3 Explorer"
Cohesion: 0.12
Nodes (24): Build a long-form editor containing only known-future covariate cells., scenario_template(), imputation_preview(), DataFrame, Show missing context cells and model interpolation, excluding holdout targets., execute_forecast(), ForecastSettings, prepare_batch() (+16 more)

### Community 34 - "Workbench Artifacts"
Cohesion: 0.17
Nodes (22): clean_retention(), preview_retention(), make_artifact_store(), S3 conditional puts retain the same immutable-key contract as local files., S3ArtifactStore, apply_retention(), _artifact_keys(), cleanup_orphans() (+14 more)

### Community 35 - "TimesFM Timesfm 2P5 Torch"
Cohesion: 0.11
Nodes (16): Path, PyTorchModelHubMixin, PyTorch implementation of TimesFM 2.5 with 200M parameters., Loads a TimesFM model from a checkpoint directory or file., Loads a PyTorch safetensors TimesFM model from a local path or the Hugging Face…, Saves the model's state dictionary to a safetensors file. This method is called…, TimesFM_2p5_200M_torch, _flip_quantile() (+8 more)

### Community 36 - "Forecasting Examples Finetuning"
Cohesion: 0.13
Nodes (16): Dataset, Namespace, evaluate(), load_retail_sales(), main(), parse_args(), ndarray, Validation dataset using the last window of each series. (+8 more)

### Community 37 - "Tests App Native"
Cohesion: 0.17
Nodes (18): psutil, subprocess, fixture, skipif, Native ownership regressions using only disposable processes created here., records(), spawned(), test_cancellation_never_terminates_an_unmatched_worker() (+10 more)

### Community 38 - "Workbench Schemas"
Cohesion: 0.11
Nodes (14): model_validator, run_replay(), Saved-input displays and summaries, independent of model loading., Use saved overall scores and complete forecast pairs, not display samples., replay_script(), summarize(), selected(), RunSpec (+6 more)

### Community 39 - "Tests Analysis Extensions"
Cohesion: 0.16
Nodes (17): configurations(), Predictor, Any, MonkeyPatch, ndarray, parametrize, Paired configuration comparisons, naive references, and interval summaries., test_comparisons_require_full_warmup_for_every_window() (+9 more)

### Community 40 - "Tests App Services"
Cohesion: 0.12
Nodes (12): execute(), Parity and process-boundary checks for the durable application shell., test_frozen_local_model_survives_source_mutation(), test_gpu_lock_excludes_second_process_without_loading_torch(), test_holdout_point_only_parity_preserves_missing_actuals(), test_input_limits_stop_decoding_before_remaining_versions(), test_preview_never_resolves_model_and_is_json_safe(), test_refresh_restores_saved_calendar_and_forecast_settings() (+4 more)

### Community 41 - "Forecasting Examples Covariates Forecasting"
Cohesion: 0.09
Nodes (21): Actual sales (Store A), Baseline (no covariates), Future event schedules must be known for XReg, Holiday effect (+200 units), Holiday weeks: +256 units avg, Price Covariate -- Context + Forecast Horizon, Price effect (max +/-10 units), Price elasticity: -$1 increase -> -20 units sold (+13 more)

### Community 42 - "TimesFM Xreg Lib"
Cohesion: 0.14
Nodes (16): BatchedInContextXRegBase, BatchedInContextXRegLinear, Any, Array, Category, ndarray, Initializes with the exogenous covariate inputs. Here we use model fitting…, Verifies the validity of the covariate inputs. (+8 more)

### Community 43 - "Workbench UI Dependencies"
Cohesion: 0.10
Nodes (21): dependencies, class-variance-authority, clsx, echarts, geist, @hookform/resolvers, lucide-react, next (+13 more)

### Community 44 - "Tests App Jobs"
Cohesion: 0.13
Nodes (15): alembic, DeclarativeBase, database_url(), Alembic entry point: the application settings own database configuration., sqlalchemy, sqlalchemy_engine, sqlalchemy_orm, sqlalchemy_pool (+7 more)

### Community 45 - "TimesFM-3 Timesfm3 Forecaster"
Cohesion: 0.15
Nodes (4): object, FakeModel, A tiny fake model for testing., TimesFM3ForecasterTest

### Community 46 - "Python Dependencies Tensor"
Cohesion: 0.14
Nodes (18): get_transform(), identity(), max_output(), _max_output_identity(), _max_output_signed_log(), _max_output_signed_sqrt(), Protocol, Tensor (+10 more)

### Community 47 - "TimesFM Normalization"
Cohesion: 0.14
Nodes (11): Tensor, RMSNorm, MultiHeadAttention, TransformerConfig, RotaryPositionalEmbedding, Tests for RMS normalization used in transformer attention/FF blocks., RMSNorm must not change the tensor shape., With default scale (initialized to zeros), output must be all zeros. This is a… (+3 more)

### Community 48 - "Tests Base Utils"
Cohesion: 0.10
Nodes (11): Tests for linear_interpolation — fills NaN gaps via ``np.interp``., Without NaN values the array is returned as-is (fast path)., A single interior NaN is linearly interpolated from neighbors., Multiple consecutive interior NaN values are interpolated., Trailing NaN values are filled via ``np.interp`` which holds the last known…, Leading NaN values are filled with the first valid value. In practice…, After interpolation, no NaN values should remain., Non-NaN values in the original array must never be modified. (+3 more)

### Community 49 - "Tests Diagnostic Client"
Cohesion: 0.15
Nodes (16): ast, export(), jobs(), download(), Path, The temporary diagnostic UI uses HTTP and never acquires a model., response(), test_cancel_and_retry_use_api() (+8 more)

### Community 50 - "Documentation Diagrams"
Cohesion: 0.16
Nodes (19): Cancelled, Cancelling, Failed, Interruptions, Job execution, Queued, Retryable, Running (+11 more)

### Community 51 - "TimesFM-3 Model"
Cohesion: 0.16
Nodes (13): no_grad, Any, PyTorchModelHubMixin, ResidualBlockConfig, StackedTransformersConfig, Tensor, Returns a serializable dictionary of model configuration., Saves model weights and config to a local directory or pushes to Hub. (+5 more)

### Community 52 - "TimesFM Util"
Cohesion: 0.15
Nodes (11): Reversible instance normalization. `mu`/`sigma` are broadcast onto `x` by…, revin(), Tests for the RevIN normalization used in patched decoding., normalize → denormalize must reconstruct the original tensor. This is the…, After forward normalization: (x - mu) / sigma., After reverse: x * sigma + mu., When sigma < tolerance, the function substitutes 1.0 to avoid division by zero.…, Sigma values just below ``_TOLERANCE`` must trigger the guard. (+3 more)

### Community 53 - "Workbench UI Tsconfig"
Cohesion: 0.11
Nodes (18): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+10 more)

### Community 54 - "Documentation Timesfm3 Guide"
Cohesion: 0.12
Nodes (18): Architecture and Data Flow, BatchPredictor Protocol, Explorer DuckDB Store, TimesFM3Torch, TimesFM-3 Explorer, How to Use the TimesFM-3 Python API, ForecastOutput, ModelConfig (+10 more)

### Community 55 - "Python Dependencies Pathlib"
Cohesion: 0.15
Nodes (11): json, nbformat, pathlib, sys, Direct runner for executing fev_bench_timesfm3.ipynb sequentially., Direct runner for executing time_bench_timesfm3.ipynb sequentially., Generate animation data for interactive forecast visualization. This script…, # NOTE: columns below are read as [q10, q20, ..., q80, q90] (index 0 = (+3 more)

### Community 56 - "Research Knowledge Times Fm Source"
Cohesion: 0.24
Nodes (18): TimesFM-3 Evaluation Claims and Reproduction Assets, BigQuery built-in TimesFM 2.5, BigQuery ML AI.FORECAST reference, FEV-Bench leaderboard, GIFT-Eval leaderboard, Contiguous Patch Masking forecasting, Google Research TimesFM-3 launch post, TimesFM-3 multivariate architecture (+10 more)

### Community 57 - "TimesFM Configs"
Cohesion: 0.17
Nodes (12): RandomFourierFeaturesConfig, Framework-agnostic config for random fourier features., Tensor, RandomFourierFeatures, Tests for RandomFourierFeaturesConfig., TestRandomFourierFeaturesConfig, Tests for the random Fourier feature layer., Output dims must be exactly ``config.output_dims``. (+4 more)

### Community 58 - "TimesFM Timesfm 2P5 Base"
Cohesion: 0.15
Nodes (11): Removes contiguous NaN values from the beginning of a NumPy array. Args: arr:…, strip_leading_nans(), Tests for strip_leading_nans — removes leading NaN prefix., An array without NaN values must pass through unmodified., Leading NaNs are removed; NaNs embedded in the middle are kept., Edge case: exactly one leading NaN., If the first element is valid, nothing is stripped regardless of internal NaNs., A single non-NaN element must be returned as-is. (+3 more)

### Community 59 - "TimesFM Util"
Cohesion: 0.14
Nodes (11): Tensor, Updates the running stats. Merges prior running (n, mu, sigma) with the stats…, update_running_stats(), Masked positions must be completely ignored — as if they don't exist. In…, When every element is masked, the function must return zeros rather than NaN or…, Each sample in the batch must be computed independently. Cross-sample leakage…, A constant series has zero variance — sigma must be exactly 0. This is…, Tests for Welford-style online mean / variance accumulation. (+3 more)

### Community 60 - "Python Dependencies Argparse"
Cohesion: 0.14
Nodes (15): argparse, asyncio, crawl4ai, crawl4ai_async_crawler_strategy, datetime, main(), Export API contracts without connecting to PostgreSQL or loading a model., Write the model-free FastAPI OpenAPI contract to the requested path. (+7 more)

### Community 61 - "TimesFM Timesfm 2P5 Flax"
Cohesion: 0.15
Nodes (14): scan, _apply_stacked_transformers(), _create_stacked_transformers(), Array, Bool, DecodeCache, Float, StackedTransformersConfig (+6 more)

### Community 62 - "TimesFM Configs"
Cohesion: 0.15
Nodes (10): ForecastConfig, Options for forecasting. Attributes: max_context: The maximum context length.…, compiled_decode_kernel(), Tests for ForecastConfig — the primary user-facing configuration., Default config must be conservative: no normalization, no fancy heads. These…, Configs are frozen dataclasses — mutating them must raise. This is critical…, ``dataclasses.replace`` must yield a new object with updated fields. The…, Two configs with identical fields must be equal (value semantics). (+2 more)

### Community 63 - "Workbench UI Components"
Cohesion: 0.12
Nodes (16): aliases, components, hooks, lib, ui, utils, iconLibrary, rsc (+8 more)

### Community 64 - "Documentation Diagrams"
Cohesion: 0.18
Nodes (16): Inference Worker, Outbox Dispatcher, Durability: submissions persist jobs and outbox entries atomically; events and immutable run manifests persist in PostgreSQL, Execution: Dramatiq separates API requests from model execution; worker leases and heartbeats protect a running attempt, PostgreSQL, Redis / Dragonfly job queues, Dramatiq separates API requests from model execution, Events and immutable run manifests persist in PostgreSQL (+8 more)

### Community 65 - "Research Knowledge Project Knowledge Index"
Cohesion: 0.15
Nodes (16): Licensing and Model Versions, Original TimesFM Paper, Patched Multivariate Transformer, TimesFM-3 Architecture and Inference, TimesFM-3 Code and Weights Licensing Boundary, Upstream Repository License Reference, TimesFM-3 Limitations and Open Questions, TimesFM-3 Primary-source Deep Research (+8 more)

### Community 66 - "Forecasting Examples Forecast Csv"
Cohesion: 0.19
Nodes (15): forecast_series(), load_csv(), load_model(), main(), DataFrame, Load CSV and identify time series columns. Returns: (dataframe,…, Forecast all series and return results dict., End-to-end CSV forecasting with TimesFM. Loads a CSV, runs the system preflight… (+7 more)

### Community 67 - "TimesFM-3 Explorer"
Cohesion: 0.14
Nodes (13): altair, parse_upload(), Parse a CSV or Parquet upload through a short-lived temporary file., streamlit, _acquire_forecaster(), _append_run(), _numeric_candidates(), _parse_in_session() (+5 more)

### Community 68 - "Documentation Explanation"
Cohesion: 0.21
Nodes (15): Attempt Fencing, Immutable Run Manifest, React and FastAPI Workbench Architecture, Workbench Job Outbox and Dispatcher, Run the Native Windows Workbench, How to Prepare Data, Legacy TimesFM-3 Explorer Handbook, Archived TimesFM 2.5 CSV Helper Guide (+7 more)

### Community 69 - "Workbench Api"
Cohesion: 0.21
Nodes (15): clean(), chart(), dataset_frame(), dataset_plot(), dataset_preview(), dataset_profile(), filter_frame(), load_table() (+7 more)

### Community 70 - "Tests Forecast Csv"
Cohesion: 0.21
Nodes (11): importlib_util, ndarray, parametrize, Path, Tests for the standalone forecast_csv.py example script. Covers series…, RecordingModel, test_csv_output_contains_explicit_and_legacy_quantile_columns(), test_forecast_exposes_explicit_quantiles_and_legacy_aliases() (+3 more)

### Community 72 - "TimesFM-3 Transformer"
Cohesion: 0.14
Nodes (6): MaskTest, MixingTransformerTest, MultiHeadAttentionTest, Tests for TimesFM3 PyTorch transformer layers., RotaryEmbeddingTest, StackedMixingTransformerTest

### Community 73 - "Documentation Diagrams"
Cohesion: 0.17
Nodes (13): Explorer DuckDB Persistence, Explorer Local Data Sources, TimesFM-3 Explorer Architecture Diagram, Checkpoint Source, Historical Analysis, Local Data Sources, Local DuckDB, Preparation and Validation (+5 more)

### Community 74 - "TimesFM Transformer"
Cohesion: 0.23
Nodes (9): Integer, make_attn_mask(), Array, Bool, DecodeCache, Float, partial, Applies multi-head dot product attention on the input data. (+1 more)

### Community 75 - "Workbench Dataset Explorer"
Cohesion: 0.23
Nodes (11): plot_data(), profile(), DataFrame, Read-only EDA over an immutable uploaded dataset; no model is loaded., Aggregate distributions on all rows and sample paired chart points evenly., Compute full-data summaries, bounding the quadratic correlation output., EDA summaries must use the whole source, never just its display sample., test_categories_all_missing_numeric_and_non_numeric_frames() (+3 more)

### Community 76 - "Tests App Services"
Cohesion: 0.17
Nodes (12): Create an isolated SQLite store. Never used by application configuration., fixture, store(), test_cancel_callback_not_invoked_after_job_already_stopped(), inputs(), fixture, worker_store(), fixture (+4 more)

### Community 77 - "TimesFM Timesfm 2P5 Base"
Cohesion: 0.17
Nodes (10): linear_interpolation(), Category, ndarray, Abstract base class for TimesFM models. Attributes: forecast_config:…, Compiles the TimesFM model for fast decoding., Forecasts the time series., Forecasts on a list of time series with covariates. To optimize inference…, Performs linear interpolation to fill NaN values in a 1D numpy array. Args:… (+2 more)

### Community 78 - "TimesFM Timesfm 2P5 Torch"
Cohesion: 0.18
Nodes (7): DecodeCache, ndarray, Tensor, Decodes the time series., Forecasts the time series. This is a naive implementation for debugging…, Loads a PyTorch TimesFM model from a checkpoint., TimesFM_2p5_200M_torch_module

### Community 79 - "Python Dependencies Matplotlib Pyplot"
Cohesion: 0.18
Nodes (9): matplotlib, matplotlib_dates, matplotlib_pyplot, pil, create_frame(), main(), Generate animated GIF showing forecast evolution. Creates a GIF animation…, Create a single frame of the animation with fixed axes. (+1 more)

### Community 80 - "Forecasting Examples Anomaly Detection"
Cohesion: 0.27
Nodes (11): matplotlib_patches, build_synthetic_future(), detect_context_anomalies(), detect_forecast_anomalies(), main(), plot_results(), ndarray, Build a plausible future with 3 injected anomalies. Injected months: 3, 8, 11… (+3 more)

### Community 81 - "TimesFM Util"
Cohesion: 0.20
Nodes (11): Array, Bool, Float, jit, partial, Reversible per-instance normalization. `mu`/`sigma` are broadcast onto `x` by…, Updates the running stats. Merges the (n, mu, sigma) running mean/std computed…, revin() (+3 more)

### Community 82 - "Forecasting Examples Anomaly Detection"
Cohesion: 0.17
Nodes (12): 80 PI and 60 PI, Context and forecast time-series plot, Context CRITICAL, CRITICAL, WARNING, Normal, Delta from expected (C) plot, Forecast CRITICAL, Forecast WARNING, Linear trend (+4 more)

### Community 83 - "Python Dependencies Uuid"
Cohesion: 0.20
Nodes (10): Any, Temporary Streamlit diagnostic client for the local workbench API. Thin HTTP…, Report recoverable API failures without losing editor or selection state.…, Keep selection attached to a stable ID when ordering or status changes., read_records(), request_api(), select_record(), httpx (+2 more)

### Community 84 - "Documentation Diagrams"
Cohesion: 0.22
Nodes (11): Artifact Store, FastAPI, Memurai and Dramatiq Queues, React Workbench, TimesFM-3 Local Workbench Architecture Diagram, TimesFM-3 Services, Product: React owns presentation and server-state caching; the browser does not own inference or saved results, React owns presentation and server-state caching (+3 more)

### Community 85 - "Documentation Diagrams"
Cohesion: 0.29
Nodes (11): 01 / Job execution, 02 / Interruptions, 03 / Terminal exits, Cancelled — stop confirmed, Cancelling — cooperative stop, Failed — unrecoverable error, Queued — validated submission, Retryable — retry / lease (+3 more)

### Community 86 - "Workbench Config"
Cohesion: 0.22
Nodes (7): Middleware, get_settings(), _JSONFormatter, LogRecord, Acquire the device before consuming messages and report idle heartbeats., _WorkerLifecycle, report()

### Community 87 - "TimesFM Dense"
Cohesion: 0.20
Nodes (7): Residual block with two linear layers and a linear residual connection.…, Array, Float, ResidualBlockConfig, RandomFourierFeatures, Random Fourier features layer., ResidualBlock

### Community 88 - "Tests Torch Layers"
Cohesion: 0.22
Nodes (7): fixture, Unsupported activation must raise ``ValueError`` immediately — fail fast rather…, Tests for the residual block: hidden → activation → output + skip., A small residual block with SiLU/Swish activation (matches TimesFM)., Output must have the config's ``output_dims`` as the last dimension, regardless…, The block must handle (batch, seq, features) inputs — the layout used when…, TestResidualBlock

### Community 89 - "Forecasting Examples Data Preparation Md"
Cohesion: 0.18
Nodes (11): Forecast Input NaN Handling, Archived TimesFM 2.5 Data Preparation, Current Prepare Data Guide, Preparing XReg Covariates, Seasonal Cycle Context Guideline, Time Series NaN Preparation, Univariate NumPy Time Series Inputs, Archived TimesFM 2.5 System Requirements (+3 more)

### Community 90 - "TimesFM Util"
Cohesion: 0.24
Nodes (8): register_dataclass, DecodeCache, Cache for decoding. `key`/`value` are pre-allocated to the full cache length…, DecodeCache, Tests for the DecodeCache dataclass used in KV-cache decoding., DecodeCache is *not* frozen — the attention loop mutates ``next_index`` and…, Key and value tensors must have identical shapes — they are indexed in parallel…, TestDecodeCache

### Community 91 - "Tests Configs"
Cohesion: 0.27
Nodes (8): Framework-agnostic config for a stacked transformers., Framework-agnostic config for a transformer., StackedTransformersConfig, TransformerConfig, Tests for TransformerConfig — architecture-level hyperparameters., The model instantiation will fail if this invariant is broken. We verify the…, StackedTransformersConfig must wrap a TransformerConfig cleanly., TestTransformerConfig

### Community 92 - "TimesFM Configs"
Cohesion: 0.24
Nodes (7): Framework-agnostic config for a residual block., ResidualBlockConfig, Tests for ResidualBlockConfig used by tokenizer and output projections., All three activation modes must be constructable without error., TestResidualBlockConfig, parametrize, All supported activations must produce finite, non-NaN output.

### Community 93 - "TimesFM Normalization"
Cohesion: 0.22
Nodes (6): RMSNorm, MultiHeadAttention, TransformerConfig, Multi-head attention., Rotary positional embedding., RotaryPositionalEmbedding

### Community 94 - "Python Dependencies Tensor"
Cohesion: 0.24
Nodes (7): Generates a JTensor of sinusoids with different frequencies., Classic Transformer used in TimesFM. Uses "sandwich" normalization per sub-…, Transformer, make_attn_mask(), DecodeCache, Tensor, Transformer

### Community 95 - "Forecasting Examples Covariates Forecasting"
Cohesion: 0.29
Nodes (8): create_visualization(), demonstrate_api(), explain_xreg_modes(), generate_sales_data(), main(), 2x2 figure -- ALL panels share x-axis = weeks 0-35. (0,0) Sales by store --…, TimesFM Covariates (XReg) Example Demonstrates the TimesFM covariate API using…, Generate synthetic retail sales data with covariate components stored…

### Community 96 - "Forecasting Examples Global Temperature"
Cohesion: 0.20
Nodes (10): 60% CI, 80% CI, Forecast period from 2025-01 to 2026-01, Forecast visualization image, Historical (NOAA GISTEMP), Mean forecast: 1.24°C, Temperature anomaly (°C), TimesFM Forecast (+2 more)

### Community 97 - "Workbench UI Dev Dependencies"
Cohesion: 0.20
Nodes (10): devDependencies, openapi-typescript, @playwright/test, prettier, @types/node, @types/react, @types/react-dom, typescript (+2 more)

### Community 98 - "Documentation Codebase"
Cohesion: 0.36
Nodes (9): Codebase Architecture, DuckDB Retention and Session-Only Inputs, Legacy Explorer Forecast Flow, make_run_artifact, parse_upload, prepare_batch, Primary React and FastAPI Workbench, resolve_model (+1 more)

### Community 99 - "Project Review"
Cohesion: 0.22
Nodes (9): DuckDB History and Session Fallback, Troubleshooting Guide, DuckDB Derived Run Retention, DuckDB Documentation Sync Report, Session-Scoped Upload Data, Atomic Upload Parsing, Code Review Findings, Exact Application Process Ownership (+1 more)

### Community 100 - "TimesFM Dense"
Cohesion: 0.22
Nodes (4): ResidualBlockConfig, ResidualBlock, Gradients must reach both the main path and the residual path. Dead gradients…, The residual connection must contribute to the output. We verify this by…

### Community 101 - "Tests App Services"
Cohesion: 0.22
Nodes (6): FakePredictor, Test-only deterministic predictor; the application always loads real weights., test_cache_switch_releases_before_loading(), load(), test_cancellation_stops_between_analysis_calls_before_artifacts(), test_disabled_signals_change_preview_and_execution_without_losing_roles()

### Community 102 - "Forecasting Examples Global Temperature"
Cohesion: 0.22
Nodes (9): All observed data, Data used, Final forecast, Forecast, Forecast from 2022-12, Shaded forecast band, Temperature Anomaly (°C) time series, TimesFM forecast evolution animation (+1 more)

### Community 103 - "Workbench UI Readme"
Cohesion: 0.31
Nodes (9): FastAPI Native Service API, Generated TypeScript API Types, Immutable Run Submission Snapshot, Native Python TimesFM Service, Next.js Analytical Workspace, Persisted Forecast Calibration Results, TanStack Query, TimesFM Analytical Workspace README (+1 more)

### Community 104 - "Documentation Readme"
Cohesion: 0.32
Nodes (8): Coding Conventions, External Integrations, Memurai and Dramatiq Job Delivery, PostgreSQL Durable Workbench State, Technology Stack, Codebase Structure, Testing Patterns, TimesFM-3 Documentation Map

### Community 105 - "Documentation Diagrams"
Cohesion: 0.32
Nodes (8): Artifact store (local files or S3), FastAPI (validation, SSE, port 8001), Inference worker (GPU forecasts, CPU checks), Memurai / Dramatiq job queues, Outbox dispatcher (durable delivery), PostgreSQL (records, outbox, events), React workbench (Next.js, port 3000), TimesFM-3 services (forecast, analysis, tracking)

### Community 106 - "TimesFM-3 Dense"
Cohesion: 0.29
Nodes (5): ResidualBlockConfig, Tensor, Forward pass. x shape: (b, ..., input_dim)., Reinitialize linear layers with the correct input dimension., ResidualBlock

### Community 107 - "TimesFM Timesfm 2P5 Flax"
Cohesion: 0.25
Nodes (5): Loads a TimesFM model from a checkpoint., Path, Flax implementation of TimesFM 2.5 with 200M parameters., Loads a Flax TimesFM model., TimesFM_2p5_200M_flax

### Community 108 - "TimesFM-3 Benchmarks Fev Bench"
Cohesion: 0.32
Nodes (8): FEV-Bench Evaluation Guide, GIFT-Eval Evaluation Guide, AutoGluon FEV-Bench, ICML TIME Benchmark, Salesforce GIFT-Eval, TimesFM 3.0, TimesFM 3.0 Benchmarks Overview, TIME Benchmark Evaluation Guide

### Community 109 - "Workbench UI Scripts"
Cohesion: 0.25
Nodes (8): scripts, build, dev, generate:api, start, test, test:e2e, typecheck

### Community 110 - "Workbench UI Workspace Spec"
Cohesion: 0.29
Nodes (7): dataset, fixture(), forecastRows, record(), run, spec, version

### Community 111 - "Workbench Worker"
Cohesion: 0.29
Nodes (5): GPUProcessLock, _own_gpu(), Path, For tests/shutdown only; production retains the handle until process exit., An OS-owned handle released on process death, including after PC sleep.

### Community 112 - "TimesFM Normalization"
Cohesion: 0.33
Nodes (4): LayerNorm, Array, Float, Layer normalization replica of LayerNorm.

### Community 114 - "TimesFM-3 Normalization"
Cohesion: 0.33
Nodes (4): PerDimScale, Tensor, Per-dimension scaling (Pax-style). Replaces the standard 1/sqrt(d) query…, Applies per-dim scaling to the last dimension of x.

### Community 115 - "TimesFM Transformer"
Cohesion: 0.33
Nodes (3): PerDimScale, Per-dimension scaling., PerDimScale

### Community 116 - "Forecasting Examples Api Reference Md"
Cohesion: 0.53
Nodes (6): Archived TimesFM 2.5 API Reference, Forecast with Covariates API, ForecastConfig, TimesFM 2.5 PyTorch Model, TimesFM Forecast API, XReg Covariates

### Community 117 - "Workbench UI Lib"
Cohesion: 0.33
Nodes (5): components, $defs, operations, paths, webhooks

### Community 118 - "Repository Automation Copilot Instructions"
Cohesion: 0.40
Nodes (5): TimesFM Agent Entry Point, Claude Instructions, Gemini Instructions, Copilot Instructions, Open Knowledge Format v0.2

### Community 119 - "Project Readme"
Cohesion: 0.40
Nodes (5): Codebase Concerns, Default Weights Commercial and Production Restriction, Original TimesFM Paper, TimesFM-3 Non-Commercial Weights License, TimesFM-3 Workbench README

### Community 120 - "Research Knowledge Timesfm3 Api And"
Cohesion: 0.50
Nodes (5): TimesFM-3 Python API and Input Shapes, TimesFM-3 forecaster API and tests, TimesFM upstream repository baseline, Archived TimesFM 2.5 compatibility skill, TimesFM-3 recommended interfaces

### Community 121 - "Tests Base Utils"
Cohesion: 0.40
Nodes (4): Tests for NaN-handling and interpolation utilities in the base module.…, Batch padding must remain internal to the forecasting call., test_forecast_does_not_mutate_input_list(), timesfm_timesfm_2p5_timesfm_2p5_base

### Community 122 - "Documentation Diagrams"
Cohesion: 0.50
Nodes (4): Durable submission, job and outbox entry, The job and outbox entry are created atomically, Validation precedes queueing

### Community 125 - "Tests App Services"
Cohesion: 0.50
Nodes (3): parametrize, test_service_preserves_core_predictions_and_exports(), test_worker_retry_policy_uses_durable_store()

### Community 126 - "Workbench UI Next Env D"
Cohesion: 0.50
Nodes (3): NOTE: This file should not be edited, web_next_types_root_params_d, web_next_types_routes_d

### Community 127 - "Documentation Diagrams"
Cohesion: 0.67
Nodes (3): Heartbeats, lease, model work

### Community 128 - "Documentation Diagrams"
Cohesion: 0.67
Nodes (3): Recovery, Cancellation and unrecoverable failure are terminal exits, Expired leases can return to the queue

### Community 129 - "Project Prometheus"
Cohesion: 0.67
Nodes (3): Prometheus TimesFM scrape configuration, TimesFM API metrics scrape job, TimesFM GPU and CPU worker metrics scrape jobs

### Community 130 - "Project Requirements"
Cohesion: 0.67
Nodes (3): Generated Python Requirements, huggingface-hub Dependency, safetensors Dependency

### Community 131 - "TimesFM-3 Tracking"
Cohesion: 0.67
Nodes (3): DatetimeIndex, Series, _timestamps()

### Community 134 - "Forecasting Examples Global Temperature"
Cohesion: 0.67
Nodes (3): TimesFM interactive forecast evolution animation, Archived TimesFM global temperature anomaly forecast report, NOAA GISTEMP Global Temperature Anomaly data

## Ambiguous Edges - Review These
- `Outbox Dispatcher` → `TimesFM-3 Services`  [AMBIGUOUS]
  docs/diagrams/workbench-architecture.visual-check.1440x900.light.png · relation: calls
- `Outbox Dispatcher` → `PostgreSQL`  [AMBIGUOUS]
  docs/diagrams/workbench-architecture.visual-check.2048x1320.dark.png · relation: shares_data_with
- `Retryable — retry / lease` → `Running — claimed attempt`  [AMBIGUOUS]
  docs/diagrams/workbench-lifecycle.lifecycle.visual-check.2048x1320.light.png · relation: conceptually_related_to

## Knowledge Gaps
- **266 isolated node(s):** `Role`, `CleanupPreview`, `Policy`, `components`, `$defs` (+261 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1078 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **35 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Outbox Dispatcher` and `TimesFM-3 Services`?**
  _Edge tagged AMBIGUOUS (relation: calls) - confidence is low._
- **What is the exact relationship between `Outbox Dispatcher` and `PostgreSQL`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **What is the exact relationship between `Retryable — retry / lease` and `Running — claimed attempt`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `create_app()` connect `Workbench Api` to `Workbench Config`, `TimesFM-3 Explorer`, `Workbench Artifacts`, `TimesFM-3 Explorer`, `TimesFM-3 Model Loading`, `Workbench Store`, `Workbench Schemas`, `Workbench Api`, `Workbench Artifacts`, `Workbench Dataset Explorer`, `Tests Diagnostic Client`, `Workbench Schemas`, `Workbench Config`, `Python Dependencies Os`, `Python Dependencies Argparse`, `Workbench Store`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `Store` connect `Workbench Store` to `Workbench Config`, `Workbench Artifacts`, `Workbench Artifacts`, `Workbench Api`, `Tests App Jobs`, `Tests App Jobs`, `Tests App Services`, `Workbench Schemas`, `Workbench Worker`, `Workbench Native`, `Workbench Migration`, `Workbench Config`, `Workbench Store`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Why does `ExplorerError` connect `TimesFM-3 Explorer` to `TimesFM-3 Run Store`, `TimesFM-3 Explorer`, `TimesFM-3 Model Loading`, `Python Dependencies Pandas`, `TimesFM-3 Explorer`, `TimesFM-3 Tracking`, `Tests Analysis Extensions`, `TimesFM-3 Analysis Ui`, `TimesFM-3 Data Preparation`, `Workbench Worker`, `Workbench Services`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `create_app()` (e.g. with `lifespan()` and `Settings`) actually correct?**
  _`create_app()` has 15 INFERRED edges - model-reasoned connections that need verification._