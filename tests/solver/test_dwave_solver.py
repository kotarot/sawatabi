# Copyright 2020 Kotaro Terada
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

import pytest

from sawatabi.model import LogicalModel
from sawatabi.solver import DWaveSolver


def test_dwave_default_solver_name():
    solver = DWaveSolver()
    assert solver._solver == "Advantage_system1.1"


def test_dwave_solver_with_logical_model_fails():
    model = LogicalModel(mtype="ising")
    solver = DWaveSolver()
    with pytest.raises(TypeError):
        solver.solve(model)


def test_dwave_solver_with_empty_model_fails():
    model = LogicalModel(mtype="ising")
    physical = model.to_physical()
    solver = DWaveSolver()
    with pytest.raises(ValueError):
        solver.solve(physical)
