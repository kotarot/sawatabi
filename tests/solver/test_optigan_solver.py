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

import os

import dimod
import numpy as np
import pytest

from sawatabi.model import LogicalModel
from sawatabi.solver import OptiganSolver


class ResponseMock:
    def __init__(self):
        self.status_code = 200

    def json(self):
        return {
            "execution_parameters": {"num_unit_steps": 10},
            "execution_time": {"annealing_time": 123.456789, "queue_time": 123.456789, "cpu_time": 123.456789, "time_stamps": [123.456798]},
            "energies": [-1.0],
            "spins": [[1, 0]],
        }


@pytest.fixture
def physical():
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction((x[0], x[1]), coefficient=-1.0)
    physical = model.to_physical()
    return physical


def test_optigan_solver(mocker, physical):
    directory = os.path.dirname(__file__)
    solver = OptiganSolver(config=f"{directory}/.optigan.yml")

    response_mock = ResponseMock()
    mocker.patch("requests.post", return_value=response_mock)

    resultset = solver.solve(physical, duplicate=True)

    assert isinstance(resultset, dimod.SampleSet)
    assert isinstance(resultset.info, dict)
    assert resultset.info["energies"] == [-1.0]
    assert resultset.info["spins"] == [[1, 0]]
    assert isinstance(resultset.variables, dimod.variables.Variables)
    assert resultset.variables == ["x[0]", "x[1]"]
    assert isinstance(resultset.record, np.recarray)
    assert np.array_equal(resultset.record[0].sample, [1, 0])
    assert resultset.record[0].energy == -1.0
    assert resultset.record[0].num_occurrences == 1
    assert isinstance(resultset.vartype, dimod.Vartype)
    assert resultset.vartype == dimod.BINARY
    assert resultset.first.sample == {"x[0]": 1, "x[1]": 0}
    assert resultset.first.energy == -1.0
    assert resultset.first.num_occurrences == 1
    for sample in resultset.samples():
        assert sample == {"x[0]": 1, "x[1]": 0}


def test_optigan_solver_without_gzip(mocker, physical):
    directory = os.path.dirname(__file__)
    solver = OptiganSolver(config=f"{directory}/.optigan.yml")

    response_mock = ResponseMock()
    mocker.patch("requests.post", return_value=response_mock)

    resultset = solver.solve(physical, gzip_request=False, gzip_response=False)
    assert isinstance(resultset, dimod.SampleSet)


def test_optigan_solver_with_invalid_response(mocker, physical):
    directory = os.path.dirname(__file__)
    solver = OptiganSolver(config=f"{directory}/.optigan.yml")

    response_mock = ResponseMock()
    response_mock.status_code = 400
    mocker.patch("requests.post", return_value=response_mock)

    with pytest.raises(ValueError):
        solver.solve(physical)


def test_optigan_solver_with_logical_model_fails():
    model = LogicalModel(mtype="ising")
    solver = OptiganSolver()
    with pytest.raises(TypeError):
        solver.solve(model)


def test_optigan_solver_with_empty_model_fails():
    model = LogicalModel(mtype="ising")
    physical = model.to_physical()
    solver = OptiganSolver()
    with pytest.raises(ValueError):
        solver.solve(physical)


def test_optigan_solver_with_ising_model_fails():
    model = LogicalModel(mtype="ising")
    s = model.variables("s", shape=(2,))
    model.add_interaction((s[0], s[1]), coefficient=-1.0)
    physical = model.to_physical()
    solver = OptiganSolver()
    with pytest.raises(ValueError):
        solver.solve(physical)


def test_optigan_solver_with_empty_config_fails():
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(2,))
    model.add_interaction((x[0], x[1]), coefficient=-1.0)
    physical = model.to_physical()
    solver = OptiganSolver(config="/tmp/.optigan.yml")
    with pytest.raises(FileNotFoundError):
        solver.solve(physical)
