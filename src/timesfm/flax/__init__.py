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

"""Flax/JAX layer implementations of TimesFM (dense, normalization, transformer).

Mirrors the `torch/` package's module set field-for-field so both backends
can be driven by the same configs in `../configs.py`. See
`transformer.py` for the top-level stack and `../timesfm_2p5/timesfm_2p5_flax.py`
for how these layers are assembled into a full model.
"""
