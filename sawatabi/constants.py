# Copyright 2021 Kotaro Terada
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

# Model Types
MODEL_ISING = "ising"
MODEL_QUBO = "qubo"

# Interaction Body Types
INTERACTION_LINEAR = 1  # 1-body
INTERACTION_QUADRATIC = 2  # 2-body

# Default label for Constraints
DEFAULT_LABEL_N_HOT = "Default N-hot Constraint"
DEFAULT_LABEL_EQUALITY = "Default Equality Constraint"
DEFAULT_LABEL_0_OR_1_HOT = "Default Zero-or-One-hot Constraint"

# Select format
SELECT_SERIES = "series"
SELECT_DICT = "dict"

# Algorithms
ALGORITHM_ATTENUATION = "attenuation"
ALGORITHM_DELTA = "delta"
ALGORITHM_INCREMENTAL = "incremental"
ALGORITHM_PARTIAL = "partial"
ALGORITHM_WINDOW = "window"

# Pick-up mode for Sawatabi Solver
PICKUP_MODE_RANDOM = "random"
PICKUP_MODE_SEQUENTIAL = "sequential"
