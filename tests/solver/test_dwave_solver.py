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

import dimod
import numpy as np
import pytest

from sawatabi.model import LogicalModel
from sawatabi.solver import DWaveSolver


@pytest.fixture
def physical():
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction((x[0], x[1]), coefficient=-1.0)
    physical = model.to_physical()
    return physical


def test_dwave_solver(mocker, physical):
    solver = DWaveSolver(solver="Advantage_system1.1")

    sampleset = dimod.SampleSet.from_samples([{"x[0]": 1, "x[1]": -1}], dimod.SPIN, energy=[-2.0])
    sampleset._info = {
        "timing": {"qpu_sampling_time": 12345},
        "problem_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    }

    mocker.patch("dwave.system.samplers.DWaveSampler.__init__", return_value=None)
    mocker.patch("dwave.system.composites.EmbeddingComposite.__init__", return_value=None)
    mocker.patch("dwave.system.composites.EmbeddingComposite.sample", return_value=sampleset)
    resultset = solver.solve(physical)

    assert isinstance(resultset, dimod.SampleSet)
    assert isinstance(resultset.info, dict)
    assert "timing" in resultset.info
    assert "problem_id" in resultset.info
    assert isinstance(resultset.variables, dimod.variables.Variables)
    assert resultset.variables == ["x[0]", "x[1]"]
    assert isinstance(resultset.record, np.recarray)
    assert np.array_equal(resultset.record[0].sample, [1, -1])
    assert resultset.record[0].energy == -2.0
    assert resultset.record[0].num_occurrences == 1
    assert isinstance(resultset.vartype, dimod.Vartype)
    assert resultset.vartype == dimod.SPIN
    assert resultset.first.sample == {"x[0]": 1, "x[1]": -1}
    assert resultset.first.energy == -2.0
    assert resultset.first.num_occurrences == 1
    for sample in resultset.samples():
        assert sample == {"x[0]": 1, "x[1]": -1}


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
