"""EDA summaries must use the whole source, never just its display sample."""

import numpy as np
import pandas as pd

from timesfm_app.api import clean
from timesfm_app.dataset_explorer import plot_data, profile


def test_profile_counts_missing_duplicates_and_finite_statistics():
  frame = pd.DataFrame(
    {
      "sales": [1.0, 3.0, np.nan, np.inf, 1.0],
      "group": ["a", "b", None, "c", "a"],
      "constant": [2, 2, 2, 2, 2],
    }
  )
  original = frame.copy(deep=True)
  result = clean(profile(frame))
  assert result["missing_cells"] == 2
  assert result["duplicate_rows"] == 1
  assert result["columns"][0]["infinite"] == 1
  assert result["columns"][0]["missing_percent"] == 20
  assert result["statistics"][0]["count"] == 3
  assert result["statistics"][0]["mean"] == 5 / 3
  assert result["correlations"][0]["constant"] is None
  pd.testing.assert_frame_equal(frame, original)


def test_plot_sample_preserves_pairs_and_histogram_uses_all_rows():
  frame = pd.DataFrame({"x": range(5001), "y": np.arange(5001) * 2})
  result = plot_data(frame, "y", "x")
  assert result["sampled"]
  assert len(result["points"]) == 2000
  assert result["points"][-1] == {"x": 5000, "y": 10000}
  assert all(p["y"] == p["x"] * 2 for p in result["points"])
  assert sum(bin["count"] for bin in result["histogram"]) == 5001


def test_categories_all_missing_numeric_and_non_numeric_frames():
  frame = pd.DataFrame({"name": ["a", "a", None], "empty": [np.nan] * 3})
  result = plot_data(frame, "name")
  assert not result["numeric"]
  assert result["categories"] == [{"value": "a", "count": 2}]
  assert result["non_null"] == 2
  assert plot_data(frame, "empty")["histogram"] == []
  assert profile(frame[["name"]])["statistics"] == []
  assert profile(frame[["name"]])["correlations"] == []


def test_correlation_columns_are_bounded_but_statistics_cover_all():
  result = profile(pd.DataFrame({f"c{i}": [1, 2, 3] for i in range(40)}))
  assert len(result["correlation_columns"]) == 32
  assert len(result["statistics"]) == 40
