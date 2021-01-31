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
from sawatabi.model.constraint import NHotConstraint
from sawatabi.solver import SawatabiSolver


def test_sawatabi_solver_ising():
    model = LogicalModel(mtype="ising")
    s = model.variables("s", shape=(2,))
    model.add_interaction(s[0], coefficient=1.0)
    model.add_interaction(s[1], coefficient=2.0)
    model.add_interaction((s[0], s[1]), coefficient=-3.0)
    model.offset(10.0)

    solver = SawatabiSolver()
    resultset = solver.solve(model.to_physical(), num_reads=2, num_sweeps=100, num_coolings=10, cooling_rate=0.8, initial_temperature=10.0, seed=12345)

    assert resultset.variables == ["s[0]", "s[1]"]
    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0].sample, [-1, 1])
    assert resultset.record[0].energy == 6.0
    assert resultset.record[0].num_occurrences == 2


def test_sawatabi_solver_qubo():
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=2.0)
    model.add_interaction((x[0], x[1]), coefficient=-5.0)
    model.offset(10.0)

    solver = SawatabiSolver()
    resultset = solver.solve(model.to_physical(), num_sweeps=1000, num_coolings=101, seed=12345)

    assert resultset.variables == ["x[0]", "x[1]"]
    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0].sample, [0, 1])
    assert resultset.record[0].energy == 8.0
    assert resultset.record[0].num_occurrences == 1


@pytest.mark.parametrize("n,s", [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 100), (10, 100)])
def test_sawatabi_solver_n_hot_ising(n, s):
    # n out of s spins should be +1
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(s,))
    model.add_constraint(NHotConstraint(variables=x, n=n))

    solver = SawatabiSolver()
    resultset = solver.solve(model.to_physical(), seed=12345)

    result = np.array(resultset.record[0].sample)
    assert np.count_nonzero(result == 1) == n
    assert np.count_nonzero(result == -1) == s - n

    # Execution time should be within practical seconds (20 sec).
    assert resultset.info["timing"]["execution_sec"] <= 20.0


@pytest.mark.parametrize("n,s", [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 100), (10, 100)])
def test_sawatabi_solver_n_hot_qubo(n, s):
    # n out of s variables should be 1
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(s,))
    model.add_constraint(NHotConstraint(variables=x, n=n))

    solver = SawatabiSolver()
    resultset = solver.solve(model.to_physical(), seed=12345)

    result = np.array(resultset.record[0].sample)
    assert np.count_nonzero(result == 1) == n
    assert np.count_nonzero(result == 0) == s - n

    # Execution time should be within practical seconds (20 sec).
    assert resultset.info["timing"]["execution_sec"] <= 20.0


@pytest.mark.parametrize("n,s,i", [(1, 4, 0), (1, 4, 1), (1, 4, 2), (1, 4, 3), (2, 10, 5)])
def test_sawatabi_solver_n_hot_ising_with_deleting(n, s, i):
    # n out of (s - 1) variables should be 1
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(s,))
    model.add_constraint(NHotConstraint(variables=x, n=n))
    model.delete_variable(x[i])

    solver = SawatabiSolver()
    resultset = solver.solve(model.to_physical(), seed=12345)

    result = np.array(resultset.record[0].sample)
    assert np.count_nonzero(result == 1) == n
    assert np.count_nonzero(result == -1) == s - n - 1


@pytest.mark.parametrize("n,s,i,j", [(1, 4, 0, 1), (1, 4, 2, 3), (2, 10, 5, 6)])
def test_sawatabi_solver_n_hot_qubo_with_deleting(n, s, i, j):
    # n out of (s - 2) variables should be 1
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(s,))
    model.add_constraint(NHotConstraint(variables=x, n=n))
    model.delete_variable(x[i])
    model.delete_variable(x[j])

    solver = SawatabiSolver()
    resultset = solver.solve(model.to_physical(), seed=12345)

    result = np.array(resultset.record[0].sample)
    assert np.count_nonzero(result == 1) == n
    assert np.count_nonzero(result == 0) == s - n - 2


def test_sawatabi_solver_ising_without_active_var():
    model = LogicalModel(mtype="ising")
    s = model.variables("s", shape=(2, 2))
    model.add_interaction(s[0, 1], coefficient=1.0)
    model.add_interaction((s[1, 0], s[1, 1]), coefficient=2.0)

    physical = model.to_physical()
    assert len(physical._label_to_index) == 3
    assert len(physical._index_to_label) == 3

    solver = SawatabiSolver()
    resultset = solver.solve(physical, seed=12345)

    assert resultset.variables == ["s[0][1]", "s[1][0]", "s[1][1]"]
    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0].sample, [1, 1, 1])
    assert resultset.record[0].energy == -3
    assert resultset.record[0].num_occurrences == 1


def test_sawatabi_solver_qubo_without_active_var():
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(2, 2))
    model.add_interaction(x[0, 1], coefficient=1.0)
    model.add_interaction((x[1, 0], x[1, 1]), coefficient=2.0)

    physical = model.to_physical()
    assert len(physical._label_to_index) == 3
    assert len(physical._index_to_label) == 3

    solver = SawatabiSolver()
    resultset = solver.solve(physical, seed=12345)

    assert resultset.variables == ["x[0][1]", "x[1][0]", "x[1][1]"]
    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0].sample, [1, 1, 1])
    assert resultset.record[0].energy == -3.0
    assert resultset.record[0].num_occurrences == 1


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


def test_sawatabi_solver_with_initial_states_ising():
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
    assert np.array_equal(resultset.record[0].sample, [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    assert resultset.record[0].energy == -12.0
    assert resultset.record[0].num_occurrences == 1


def test_sawatabi_solver_with_initial_states_qubo():
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(6,))
    for i in range(6):
        model.add_interaction(x[i], coefficient=-1.0)

    solver = SawatabiSolver()
    initial_states = [
        {
            "x[0]": 1,
            "x[1]": 0,
            "x[2]": 1,
            "x[3]": 0,
            "x[4]": 1,
            "x[5]": 0,
        },
    ]
    resultset = solver.solve(model.to_physical(), num_reads=1, num_sweeps=100, num_coolings=10, cooling_rate=0.5, initial_states=initial_states, seed=12345)

    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0].sample, [0, 0, 0, 0, 0, 0])
    assert resultset.record[0].energy == 0.0
    assert resultset.record[0].num_occurrences == 1


def test_sawatabi_solver_with_initial_states_reverse():
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(6,))
    for i in range(6):
        model.add_interaction(x[i], coefficient=10.0)

    solver = SawatabiSolver()
    initial_states = [
        {
            "x[0]": -1,
            "x[1]": -1,
            "x[2]": -1,
            "x[3]": -1,
            "x[4]": -1,
            "x[5]": -1,
        },
    ]
    resultset = solver.solve(
        model.to_physical(),
        num_reads=1,
        num_sweeps=100,
        num_coolings=10,
        initial_states=initial_states,
        reverse_options={"reverse_period": 5, "reverse_temperature": 10.0},
        seed=12345,
    )

    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0].sample, [1, 1, 1, 1, 1, 1])
    assert resultset.record[0].energy == -60.0
    assert resultset.record[0].num_occurrences == 1


def test_sawatabi_solver_when_coolings_is_larger_than_sweeps():
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(2,))
    for i in range(2):
        model.add_interaction(x[i], coefficient=-1.0)
    solver = SawatabiSolver()

    with pytest.warns(UserWarning):
        sampleset_1 = solver.solve(model.to_physical(), num_reads=1, num_sweeps=10, num_coolings=11, cooling_rate=0.5, seed=12345)
    assert np.array_equal(sampleset_1.record[0].sample, [-1, -1])
    assert sampleset_1.record[0].energy == -2.0

    with pytest.warns(UserWarning):
        sampleset_2 = solver.solve(model.to_physical(), num_reads=1, num_sweeps=10, num_coolings=20, cooling_rate=0.5, seed=12345)
    assert np.array_equal(sampleset_2.record[0].sample, [-1, -1])
    assert sampleset_2.record[0].energy == -2.0


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
