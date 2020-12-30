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

import random

import pytest

from sawatabi.model import LogicalModel
from sawatabi.utils.profile import profile


@profile
def _create_n_variable_random_complete_model(n=4, seed=None):
    if seed is not None:
        random.seed(seed)

    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(n,))
    for i in range(n):
        model.add_interaction(x[i], coefficient=random.random())
    for i in range(n - 1):
        for j in range(i + 1, n):
            model.add_interaction((x[i], x[j]), coefficient=random.random())


# TODO: Need to improve performance.
# Creation of a 100-variable fully-connected ising model should be done within seconds (which means it's not quite a long time).
@pytest.mark.parametrize("n", [10, 100])
def test_create_n_variable_random_complete_model(n):
    result = _create_n_variable_random_complete_model(n=n, seed=12345)

    # execution time should be less than 5 sec
    assert result["profile"]["elapsed_sec"] < 5.0


@profile
def _create_nxn_random_lattice_model(n=4, seed=None):
    if seed is not None:
        random.seed(seed)

    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(n, n))
    for i in range(n):
        for j in range(n):
            model.add_interaction(x[i, j], coefficient=random.random())
            if 0 < i:
                model.add_interaction((x[i - 1, j], x[i, j]), coefficient=random.random())
            if 0 < j:
                model.add_interaction((x[i, j - 1], x[i, j]), coefficient=random.random())


# TODO: Need to improve performance.
# Creation of a 50x50 lattice ising model should be done within seconds (which means it's not quite a long time).
@pytest.mark.parametrize("n", [10, 50])
def test_create_nxn_random_lattice_model(n):
    result = _create_nxn_random_lattice_model(n=n, seed=12345)

    # execution time should be less than 5 sec
    assert result["profile"]["elapsed_sec"] < 5.0
