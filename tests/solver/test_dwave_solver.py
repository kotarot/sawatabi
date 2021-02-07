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


@pytest.fixture
def default_solver(mocker):
    mocker.patch("dwave.system.samplers.DWaveSampler.__init__", return_value=None)
    mocker.patch("dwave.system.composites.EmbeddingComposite.__init__", return_value=None)

    return DWaveSolver()


def test_dwave_solver(mocker, physical):
    sampleset = dimod.SampleSet.from_samples([{"x[0]": 1, "x[1]": -1}], dimod.SPIN, energy=[-2.0])
    sampleset._info = {
        "timing": {"qpu_sampling_time": 12345},
        "problem_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    }
    mocker.patch("dwave.system.samplers.DWaveSampler.__init__", return_value=None)
    mocker.patch("dwave.system.composites.EmbeddingComposite.__init__", return_value=None)
    mocker.patch("dwave.system.composites.EmbeddingComposite.sample", return_value=sampleset)

    solver = DWaveSolver(solver="Advantage_system1.1")

    sampleset = solver.solve(physical)

    assert isinstance(sampleset, dimod.SampleSet)
    assert isinstance(sampleset.info, dict)
    assert "timing" in sampleset.info
    assert "problem_id" in sampleset.info
    assert isinstance(sampleset.variables, dimod.variables.Variables)
    assert sampleset.variables == ["x[0]", "x[1]"]
    assert isinstance(sampleset.record, np.recarray)
    assert np.array_equal(sampleset.record[0].sample, [1, -1])
    assert sampleset.record[0].energy == -2.0
    assert sampleset.record[0].num_occurrences == 1
    assert isinstance(sampleset.vartype, dimod.Vartype)
    assert sampleset.vartype == dimod.SPIN
    assert sampleset.first.sample == {"x[0]": 1, "x[1]": -1}
    assert sampleset.first.energy == -2.0
    assert sampleset.first.num_occurrences == 1
    for sample in sampleset.samples():
        assert sample == {"x[0]": 1, "x[1]": -1}


def test_dwave_solver_with_auth_parameters(mocker, physical):
    sampleset = dimod.SampleSet.from_samples([{"x[0]": 1, "x[1]": -1}], dimod.SPIN, energy=[-2.0])
    sampleset._info = {
        "timing": {"qpu_sampling_time": 12345},
        "problem_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    }
    mocker.patch("dwave.system.samplers.DWaveSampler.__init__", return_value=None)
    mocker.patch("dwave.system.composites.EmbeddingComposite.__init__", return_value=None)
    mocker.patch("dwave.system.composites.EmbeddingComposite.sample", return_value=sampleset)

    solver = DWaveSolver(endpoint="http://0.0.0.0/method", token="xxxx", solver="Advantage_system1.1")

    sampleset = solver.solve(physical)

    assert isinstance(sampleset, dimod.SampleSet)


def test_dwave_solver_with_embedding_parameters(mocker, physical):
    sampleset = dimod.SampleSet.from_samples([{"x[0]": 1, "x[1]": -1}], dimod.SPIN, energy=[-2.0])
    sampleset._info = {
        "timing": {"qpu_sampling_time": 12345},
        "problem_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    }
    mocker.patch("dwave.system.samplers.DWaveSampler.__init__", return_value=None)
    mocker.patch("dwave.system.composites.EmbeddingComposite.__init__", return_value=None)
    mocker.patch("dwave.system.composites.EmbeddingComposite.sample", return_value=sampleset)

    solver = DWaveSolver(endpoint="http://0.0.0.0/method", embedding_parameters={"random_seed": 12345})

    sampleset = solver.solve(physical)

    assert isinstance(sampleset, dimod.SampleSet)


def test_dwave_default_solver_name(default_solver):
    assert default_solver._solver == "Advantage_system1.1"


def test_dwave_solver_with_logical_model_fails(default_solver):
    model = LogicalModel(mtype="ising")
    with pytest.raises(TypeError):
        default_solver.solve(model)


def test_dwave_solver_with_empty_model_fails(default_solver):
    model = LogicalModel(mtype="ising")
    with pytest.raises(ValueError):
        default_solver.solve(model.to_physical())
