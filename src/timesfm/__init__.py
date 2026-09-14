# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TimesFM API.

Package entry point for the TimesFM v1/v2 (this package) and v2.5 model
families. Exposes `ForecastConfig` plus whichever concrete model classes
their backend (torch, flax, or the separate timesfm3 package) is installed
for. See `configs.py` for the config dataclasses and
`timesfm_2p5/timesfm_2p5_base.py` for the shared model logic that the torch
and flax variants build on.
"""

from .configs import ForecastConfig

# Each backend (torch, flax) is an optional dependency: only the ones the
# caller has installed will resolve, so import failures here are expected
# and must not abort the package import.
try:
  from .timesfm_2p5 import timesfm_2p5_torch
  TimesFM_2p5_200M_torch = timesfm_2p5_torch.TimesFM_2p5_200M_torch
except ImportError:
  pass

try:
  from .timesfm_2p5 import timesfm_2p5_flax
  TimesFM_2p5_200M_flax = timesfm_2p5_flax.TimesFM_2p5_200M_flax
except ImportError:
  pass

# timesfm3 is a separate, optionally-installed package (the v3 model line).
try:
  from timesfm3 import TimesFM3Forecaster, TimesFM3Torch
except ImportError:
  pass
