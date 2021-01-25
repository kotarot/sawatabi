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
from sawatabi.model.constraint import EqualityConstraint, NHotConstraint, ZeroOrOneHotConstraint
from sawatabi.solver import LocalSolver


def test_local_solver_exact_ising():
    model = LogicalModel(mtype="ising")
    s = model.variables("s", shape=(2,))
    model.add_interaction(s[0], coefficient=1.0)
    model.add_interaction(s[1], coefficient=2.0)
    model.add_interaction((s[0], s[1]), coefficient=-3.0)

    solver = LocalSolver(exact=True)
    resultset = solver.solve(model.to_physical())

    assert resultset.variables == ["s[0]", "s[1]"]
    assert len(resultset.record) == 4
    for r in resultset.record:
        # Check the ground state
        if np.array_equal(r.sample, [-1, 1]):
            assert r.energy == -4.0
            assert r.num_occurrences == 1
            break
    else:
        assert False


def test_local_solver_exact_qubo():
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=2.0)
    model.add_interaction((x[0], x[1]), coefficient=-5.0)

    solver = LocalSolver(exact=True)
    resultset = solver.solve(model.to_physical())

    assert resultset.variables == ["x[0]", "x[1]"]
    assert len(resultset.record) == 4
    for r in resultset.record:
        # Check the ground state
        if np.array_equal(r.sample, [0, 1]):
            assert r.energy == -2.0
            assert r.num_occurrences == 1
            break
    else:
        assert False


def test_local_solver_sa_ising():
    model = LogicalModel(mtype="ising")
    s = model.variables("s", shape=(2,))
    model.add_interaction(s[0], coefficient=1.0)
    model.add_interaction(s[1], coefficient=2.0)
    model.add_interaction((s[0], s[1]), coefficient=-3.0)
    model._offset = 10.0

    solver = LocalSolver()
    resultset = solver.solve(model.to_physical(), seed=12345)

    assert resultset.variables == ["s[0]", "s[1]"]
    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0].sample, [-1, 1])
    assert resultset.record[0].energy == 6.0
    assert resultset.record[0].num_occurrences == 1


def test_local_solver_sa_qubo():
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=2.0)
    model.add_interaction((x[0], x[1]), coefficient=-5.0)
    model._offset = 10.0

    solver = LocalSolver(exact=False)
    resultset = solver.solve(model.to_physical(), seed=12345)

    assert resultset.variables == ["x[0]", "x[1]"]
    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0].sample, [0, 1])
    assert resultset.record[0].energy == 8.0
    assert resultset.record[0].num_occurrences == 1


def test_local_solver_with_logical_model_fails():
    model = LogicalModel(mtype="ising")

    solver = LocalSolver()
    with pytest.raises(TypeError):
        solver.solve(model, seed=12345)


def test_local_solver_with_empty_model_fails():
    model = LogicalModel(mtype="ising")

    solver = LocalSolver()
    with pytest.raises(ValueError):
        solver.solve(model.to_physical(), seed=12345)


def test_local_solver_default_beta_range():
    model = LogicalModel(mtype="ising")
    s = model.variables("s", shape=(2,))
    model.add_interaction(s[0], coefficient=1.0)
    model.add_interaction(s[1], coefficient=2.0)
    model.add_interaction((s[0], s[1]), coefficient=-3.0)

    solver = LocalSolver()
    beta_range = solver.default_beta_range(model.to_physical())
    assert beta_range == [0.13862943611198905, 4.605170185988092]


def test_local_solver_default_beta_range_fails():
    model = LogicalModel(mtype="ising")

    solver = LocalSolver()
    with pytest.raises(ValueError):
        solver.default_beta_range(model.to_physical())


################################
# N-hot Constraint
################################


@pytest.mark.parametrize("n,s", [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 100), (10, 100)])
def test_local_solver_n_hot_ising(n, s):
    # n out of s spins should be +1
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(s,))
    model.add_constraint(NHotConstraint(variables=x, n=n))

    solver = LocalSolver()
    physical = model.to_physical()
    for seed in [11, 22, 33, 44, 55]:
        resultset = solver.solve(physical, seed=seed)

        result = np.array(resultset.record[0].sample)
        assert np.count_nonzero(result == 1) == n
        assert np.count_nonzero(result == -1) == s - n

        # Execution time should be within seconds (5 sec).
        assert resultset.info["timing"]["elapsed_sec"] <= 5.0
        assert resultset.info["timing"]["elapsed_counter"] <= 5.0


@pytest.mark.parametrize("n,s", [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 100), (10, 100)])
def test_local_solver_n_hot_qubo(n, s):
    # n out of s variables should be 1
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(s,))
    model.add_constraint(NHotConstraint(variables=x, n=n))

    solver = LocalSolver()
    physical = model.to_physical()
    for seed in [11, 22, 33, 44, 55]:
        resultset = solver.solve(physical, seed=seed)

        result = np.array(resultset.record[0].sample)
        assert np.count_nonzero(result == 1) == n
        assert np.count_nonzero(result == 0) == s - n

        # Execution time should be within seconds (5 sec).
        assert resultset.info["timing"]["elapsed_sec"] <= 5.0
        assert resultset.info["timing"]["elapsed_counter"] <= 5.0


@pytest.mark.parametrize("n,s,i", [(1, 4, 0), (1, 4, 1), (1, 4, 2), (1, 4, 3), (2, 10, 5)])
def test_local_solver_n_hot_ising_with_deleting(n, s, i):
    # n out of (s - 1) variables should be 1
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(s,))
    model.add_constraint(NHotConstraint(variables=x, n=n))
    model.delete_variable(x[i])

    solver = LocalSolver()
    physical = model.to_physical()
    for seed in [11, 22, 33, 44, 55]:
        resultset = solver.solve(physical, seed=seed)

        result = np.array(resultset.record[0].sample)
        assert np.count_nonzero(result == 1) == n
        assert np.count_nonzero(result == -1) == s - n - 1

        # Execution time should be within seconds (5 sec).
        assert resultset.info["timing"]["elapsed_sec"] <= 5.0
        assert resultset.info["timing"]["elapsed_counter"] <= 5.0


@pytest.mark.parametrize("n,s,i,j", [(1, 4, 0, 1), (1, 4, 2, 3), (2, 10, 5, 8)])
def test_local_solver_n_hot_qubo_with_deleting(n, s, i, j):
    # n out of (s - 2) variables should be 1
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(s,))
    model.add_constraint(NHotConstraint(variables=x, n=n))
    model.delete_variable(x[i])
    model.delete_variable(x[j])

    solver = LocalSolver()
    physical = model.to_physical()
    for seed in [11, 22, 33, 44, 55]:
        resultset = solver.solve(physical, seed=seed)

        result = np.array(resultset.record[0].sample)
        assert np.count_nonzero(result == 1) == n
        assert np.count_nonzero(result == 0) == s - n - 2

        # Execution time should be within seconds (5 sec).
        assert resultset.info["timing"]["elapsed_sec"] <= 5.0
        assert resultset.info["timing"]["elapsed_counter"] <= 5.0


################################
# Equality Constraint
################################


@pytest.mark.parametrize("m,n", [(2, 2), (10, 10), (10, 20), (50, 50)])
def test_local_solver_equality_ising(m, n):
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(m,))
    y = model.variables("y", shape=(n,))
    model.add_constraint(EqualityConstraint(variables_1=x, variables_2=y))

    solver = LocalSolver()
    physical = model.to_physical()
    for seed in [11, 22, 33, 44, 55]:
        resultset = solver.solve(physical, seed=seed)

        result = np.array(resultset.record[0].sample)
        result_1 = result[0:m]
        result_2 = result[m : (m + n)]  # noqa: E203
        assert np.count_nonzero(result_1 == 1) == np.count_nonzero(result_2 == 1)

        # Execution time should be within 5 sec.
        assert resultset.info["timing"]["elapsed_sec"] <= 5.0
        assert resultset.info["timing"]["elapsed_counter"] <= 5.0


@pytest.mark.parametrize("m,n", [(2, 2), (10, 10), (10, 20), (50, 50)])
def test_local_solver_equality_qubo(m, n):
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(m,))
    y = model.variables("y", shape=(n,))
    model.add_constraint(EqualityConstraint(variables_1=x, variables_2=y))

    solver = LocalSolver()
    physical = model.to_physical()
    for seed in [11, 22, 33, 44, 55]:
        resultset = solver.solve(physical, seed=seed)

        result = np.array(resultset.record[0].sample)
        result_1 = result[0:m]
        result_2 = result[m : (m + n)]  # noqa: E203
        assert np.count_nonzero(result_1 == 1) == np.count_nonzero(result_2 == 1)

        # Execution time should be within 5 sec...
        assert resultset.info["timing"]["elapsed_sec"] <= 5.0
        assert resultset.info["timing"]["elapsed_counter"] <= 5.0


################################
# Zero-or-One-hot Constraint
################################


@pytest.mark.parametrize("n", [2, 3, 4, 10, 100])
def test_local_solver_zero_or_one_hot_ising(n):
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(n,))
    model.add_constraint(ZeroOrOneHotConstraint(variables=x))

    solver = LocalSolver()
    physical = model.to_physical()
    for seed in [11, 22, 33, 44, 55]:
        resultset = solver.solve(physical, seed=seed)

        result = np.array(resultset.record[0].sample)
        assert np.count_nonzero(result == 1) in [0, 1]

        # Execution time should be within seconds (5 sec).
        assert resultset.info["timing"]["elapsed_sec"] <= 5.0
        assert resultset.info["timing"]["elapsed_counter"] <= 5.0


@pytest.mark.parametrize("n", [2, 3, 4, 10, 100])
def test_local_solver_zero_or_one_hot_qubo(n):
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(n,))
    model.add_constraint(ZeroOrOneHotConstraint(variables=x))

    solver = LocalSolver()
    physical = model.to_physical()
    for seed in [11, 22, 33, 44, 55]:
        resultset = solver.solve(physical, seed=seed)

        result = np.array(resultset.record[0].sample)
        assert np.count_nonzero(result == 1) in [0, 1]

        # Execution time should be within seconds (5 sec).
        assert resultset.info["timing"]["elapsed_sec"] <= 5.0
        assert resultset.info["timing"]["elapsed_counter"] <= 5.0
