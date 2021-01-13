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

import numpy as np
import pytest

from sawatabi.model import LogicalModel
from sawatabi.solver import SawatabiSolver


def test_sawatabi_solver_ising():
    model = LogicalModel(mtype="ising")
    s = model.variables("s", shape=(2,))
    model.add_interaction(s[0], coefficient=1.0)
    model.add_interaction(s[1], coefficient=2.0)
    model.add_interaction((s[0], s[1]), coefficient=-3.0)
    model._offset = 10.0
    physical = model.to_physical()
    solver = SawatabiSolver()
    resultset = solver.solve(physical, num_reads=2, num_sweeps=100, num_coolings=10, cooling_rate=0.9, initial_temperature=10.0, seed=12345)

    assert resultset.variables == ["s[0]", "s[1]"]
    assert len(resultset.record) == 2

    # Check the ground state
    assert np.array_equal(resultset.record[0][0], [-1, 1])
    assert resultset.record[0][1] == 6.0  # energy
    assert resultset.record[0][2] == 1  # num of occurrences


def test_sawatabi_solver_qubo():
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=2.0)
    model.add_interaction((x[0], x[1]), coefficient=-5.0)
    model._offset = 10.0
    physical = model.to_physical()
    solver = SawatabiSolver()
    resultset = solver.solve(physical, num_sweeps=1000, num_coolings=101, seed=12345)

    assert resultset.variables == ["x[0]", "x[1]"]
    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0][0], [0, 1])
    assert resultset.record[0][1] == 8.0  # energy
    assert resultset.record[0][2] == 1  # num of occurrences


@pytest.mark.parametrize("n,s", [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 100), (10, 100)])
def test_sawatabi_solver_n_hot_ising(n, s):
    # n out of s spins should be +1
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(s,))
    model.n_hot_constraint(x, n=n)
    physical = model.to_physical()
    solver = SawatabiSolver()
    resultset = solver.solve(physical, seed=12345)

    result = np.array(resultset.record[0][0])
    assert np.count_nonzero(result == 1) == n
    assert np.count_nonzero(result == -1) == s - n

    # Execution time should be within 5 sec.
    assert resultset.info["timing"]["elapsed_sec"] <= 5.0
    assert resultset.info["timing"]["elapsed_counter"] <= 5.0


@pytest.mark.parametrize("n,s", [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 100), (10, 100)])
def test_sawatabi_solver_n_hot_qubo(n, s):
    # n out of s variables should be 1
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(s,))
    model.n_hot_constraint(x, n=n)
    physical = model.to_physical()
    solver = SawatabiSolver()
    resultset = solver.solve(physical, seed=12345)

    result = np.array(resultset.record[0][0])
    assert np.count_nonzero(result == 1) == n
    assert np.count_nonzero(result == 0) == s - n

    # Execution time should be within 5 sec...
    assert resultset.info["timing"]["elapsed_sec"] <= 5.0
    assert resultset.info["timing"]["elapsed_counter"] <= 5.0


@pytest.mark.parametrize("n,s,i", [(1, 4, 0), (1, 4, 1), (1, 4, 2), (1, 4, 3), (2, 10, 5)])
def test_sawatabi_solver_n_hot_ising_with_deleting(n, s, i):
    # n out of (s - 1) variables should be 1
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(s,))
    model.n_hot_constraint(x, n=n)
    model.delete_variable(x[i])
    physical = model.to_physical()
    solver = SawatabiSolver()
    resultset = solver.solve(physical, seed=12345)

    result = np.array(resultset.record[0][0])
    assert np.count_nonzero(result == 1) == n
    assert np.count_nonzero(result == -1) == s - n - 1


@pytest.mark.parametrize("n,s,i,j", [(1, 4, 0, 1), (1, 4, 2, 3), (2, 10, 5, 6)])
def test_sawatabi_solver_n_hot_qubo_with_deleting(n, s, i, j):
    # n out of (s - 2) variables should be 1
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(s,))
    model.n_hot_constraint(x, n=n)
    model.delete_variable(x[i])
    model.delete_variable(x[j])
    physical = model.to_physical()
    solver = SawatabiSolver()
    resultset = solver.solve(physical, seed=12345)

    result = np.array(resultset.record[0][0])
    assert np.count_nonzero(result == 1) == n
    assert np.count_nonzero(result == 0) == s - n - 2


def test_sawatabi_solver_with_logical_model_fails():
    model = LogicalModel(mtype="ising")
    solver = SawatabiSolver()
    with pytest.raises(TypeError):
        solver.solve(model, seed=12345)


def test_sawatabi_solver_with_empty_model_fails():
    model = LogicalModel(mtype="ising")
    physical = model.to_physical()
    solver = SawatabiSolver()
    with pytest.raises(ValueError):
        solver.solve(physical, seed=12345)


def test_sawatabi_solver_with_initial_states():
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(12,))
    for i in range(12):
        model.add_interaction(x[i], coefficient=-1.0)

    solver = SawatabiSolver()
    initial_states = [
        {
            "x[0]": 1,
            "x[1]": -1,
            "x[2]": -1,
            "x[3]": -1,
            "x[4]": -1,
            "x[5]": -1,
            "x[6]": -1,
            "x[7]": -1,
            "x[8]": -1,
            "x[9]": -1,
            "x[10]": -1,
            "x[11]": -1,
        },
    ]
    resultset = solver.solve(model.to_physical(), num_reads=1, num_sweeps=1, num_coolings=1, pickup_mode="sequential", initial_states=initial_states)

    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0][0], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    assert resultset.record[0][1] == -12.0  # energy
    assert resultset.record[0][2] == 1  # num of occurrences


def test_sawatabi_solver_with_initial_states_fails():
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(2,))
    for i in range(2):
        model.add_interaction(x[i], coefficient=-1.0)
    solver = SawatabiSolver()
    initial_states = [{"x[0]": 1, "x[1]": 1}]
    with pytest.raises(ValueError):
        solver.solve(model.to_physical(), num_reads=2, initial_states=initial_states)


def test_sawatabi_solver_invalid_pickup_mode():
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(2,))
    for i in range(2):
        model.add_interaction(x[i], coefficient=-1.0)
    solver = SawatabiSolver()
    with pytest.raises(ValueError):
        solver.solve(model.to_physical(), pickup_mode="invalid")
