# Copyright 2020 Kotaro Terada, Shingo Furuyama, Junya Usui, and Kazuki Ono
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
from sawatabi.solver import LocalSolver


def test_local_solver_exact_ising():
    model = LogicalModel(mtype="ising")
    s = model.variables("s", shape=(2,))
    model.add_interaction(s[0], coefficient=1.0)
    model.add_interaction(s[1], coefficient=2.0)
    model.add_interaction((s[0], s[1]), coefficient=-3.0)
    physical = model.convert_to_physical()
    solver = LocalSolver(exact=True)
    resultset = solver.solve(physical)

    assert resultset.variables == ["s[0]", "s[1]"]
    assert len(resultset.record) == 4
    for r in resultset.record:
        # Check the ground state
        if np.array_equal(r[0], [-1, 1]):
            assert r[1] == -4.0  # energy
            assert r[2] == 1  # num of occurrences


def test_local_solver_exact_qubo():
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=2.0)
    model.add_interaction((x[0], x[1]), coefficient=-5.0)
    physical = model.convert_to_physical()
    solver = LocalSolver(exact=True)
    resultset = solver.solve(physical)

    assert resultset.variables == ["x[0]", "x[1]"]
    assert len(resultset.record) == 4
    for r in resultset.record:
        # Check the ground state
        if np.array_equal(r[0], [0, 1]):
            assert r[1] == -2.0  # energy
            assert r[2] == 1  # num of occurrences


def test_local_solver_sa_ising():
    model = LogicalModel(mtype="ising")
    s = model.variables("s", shape=(2,))
    model.add_interaction(s[0], coefficient=1.0)
    model.add_interaction(s[1], coefficient=2.0)
    model.add_interaction((s[0], s[1]), coefficient=-3.0)
    physical = model.convert_to_physical()
    solver = LocalSolver()
    resultset = solver.solve(physical)

    assert resultset.variables == ["s[0]", "s[1]"]
    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0][0], [-1, 1])
    assert resultset.record[0][1] == -4.0  # energy
    assert resultset.record[0][2] == 1  # num of occurrences


def test_local_solver_sa_qubo():
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=2.0)
    model.add_interaction((x[0], x[1]), coefficient=-5.0)
    physical = model.convert_to_physical()
    solver = LocalSolver(exact=False)
    resultset = solver.solve(physical)

    assert resultset.variables == ["x[0]", "x[1]"]
    assert len(resultset.record) == 1

    # Check the ground state
    assert np.array_equal(resultset.record[0][0], [0, 1])
    assert resultset.record[0][1] == -2.0  # energy
    assert resultset.record[0][2] == 1  # num of occurrences
