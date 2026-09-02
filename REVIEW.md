# Code Review

Reviewed: 2026-09-02. Scope: current TimesFM 3 package, Streamlit explorer,
forecasting helper, tests, launcher, and GitHub workflows. Findings are based on
the working tree and the stated explorer requirements.

## Summary

The explorer has a useful separation between Streamlit rendering and typed
forecast helpers, but the review found unsafe checkpoint loading, incomplete
batch validation, upload-memory risks, a form interaction deadlock, and unsafe
launcher/publishing behavior. The critical and warning findings below have fixes
in the current working tree; validation status is recorded separately.

## Critical issues

| Finding | Evidence | Resolution | Status |
|---|---|---|---|
| Local pickle checkpoints could execute arbitrary code | `src/timesfm3/timesfm3_forecaster.py` used unrestricted `torch.load` | Load tensor weights with `weights_only=True` | Fixed in working tree |
| Launcher killed every process listening on port 9587 | `launch_app.cmd` | Match the exact app path and port; refuse unrelated owners | Fixed in working tree |
| Fork could publish the upstream package to PyPI | `.github/workflows/manual_publish.yml` | Gate job to `google-research/timesfm` | Fixed in working tree |

## Warnings

| Finding | Evidence | Resolution | Status |
|---|---|---|---|
| Malformed batch companions failed late or produced wrong covariate padding | `TimesFM3Forecaster.predict_batch` | Validate lengths/shapes and use covariate-shaped zero templates | Fixed in working tree |
| Compressed uploads could expand beyond the raw-byte limit | `src/timesfm3/explorer.py` | Enforce per-file and aggregate decoded-memory limits | Fixed in working tree |
| Non-finite and float32-overflow values reached inference | `src/timesfm3/explorer.py`, `timesfm3_forecaster.py` | Reject upload overflow/infinity and normalize model input non-finites | Fixed in working tree |
| License checkbox inside the form could not enable its disabled submit button | `streamlit_app.py` | Move acknowledgement outside the form | Fixed in working tree |
| Multiple uploads could leave a runnable partial batch after one parse failed | `streamlit_app.py` | Parse atomically and clear the batch on any failure | Fixed in working tree |
| TimesFM 2.5 padded the caller's list in place | `src/timesfm/timesfm_2p5/timesfm_2p5_base.py` | Pad a copy | Fixed in working tree |
| CSV helper collapsed internal missing time steps | `timesfm-forecasting/scripts/forecast_csv.py` | Trim missing edges only; retain internal gaps | Fixed in working tree |

## Suggestions

- Keep the Streamlit page declarative and route forecast execution through one
  helper interface; this improves locality and makes the inference seam testable.
- Treat `v1/` as an archived, separate environment because it contains an older
  package with the same import name.
- Add an enforced coverage threshold after the Windows NumPy/coverage tooling
  incompatibility is resolved.

## Positive findings

- `DatasetMapping`, `ForecastSettings`, and `RunArtifact` make data shape and
  reproducibility explicit.
- Model caching is bounded to one heavyweight resource and user-visible failures
  avoid displaying secrets.
- The explorer already has deterministic test doubles and ZIP-manifest tests.

## Validation tracking

- Distribution build, targeted Ruff, and targeted ty: passed.
- Pytest collection: 152 tests collected without requiring Flax.
- Full local pytest: 149 passed, 1 skipped, 2 failed. Both failures are an
  existing TimesFM 2.5 CPU/CUDA placement interaction when the complete suite
  runs on a CUDA host (`test_force_flip_invariance` and the model-loading forward
  check); the application and TimesFM 3 tests are not implicated.
- Launcher: started the absolute app path on port 9587; the verified app listener
  was stopped after the check.
- The four pre-existing Git LFS media discrepancies are outside this review and
  must remain untouched.

## Verdict

Approve with the two documented upstream CUDA-host test failures. No unresolved
critical design question blocks the current repair set.
