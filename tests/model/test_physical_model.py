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

from sawatabi.model import LogicalModel, PhysicalModel


@pytest.fixture
def simple_model():
    return PhysicalModel(mtype="ising")


@pytest.fixture
def model_ising():
    model = LogicalModel(mtype="ising")
    x = model.variables(name="x", shape=(3,))
    model.delete_variable(x[0])
    model.add_interaction(x[1], coefficient=1.0)
    model.add_interaction(x[2], coefficient=2.0)
    model.add_interaction((x[1], x[2]), coefficient=3.0)
    return model.to_physical()


@pytest.fixture
def model_qubo():
    model = LogicalModel(mtype="qubo")
    x = model.variables(name="x", shape=(3,))
    model.delete_variable(x[0])
    model.add_interaction(x[1], coefficient=1.0)
    model.add_interaction(x[2], coefficient=2.0)
    model.add_interaction((x[1], x[2]), coefficient=3.0)
    return model.to_physical()


################################
# Physical Model
################################


@pytest.mark.parametrize("mtype", ["ising", "qubo"])
def test_physical_model_constructor(mtype):
    model = PhysicalModel(mtype=mtype)
    assert model.get_mtype() == mtype
    assert model._mtype == mtype


################################
# Converts to another model
################################


def test_convert_to_bqm_ising(model_ising):
    bqm = model_ising.to_bqm()
    bqm_ising = bqm.to_ising()
    # linear
    assert bqm_ising[0]["x[1]"] == -1.0
    assert bqm_ising[0]["x[2]"] == -2.0
    # quadratic
    assert bqm_ising[1][("x[1]", "x[2]")] == -3.0
    # offset
    assert bqm_ising[2] == 0.0


def test_convert_to_bqm_qubo(model_qubo):
    bqm = model_qubo.to_bqm()
    bqm_qubo = bqm.to_qubo()
    # linear
    assert bqm_qubo[0][("x[1]", "x[1]")] == -1.0
    assert bqm_qubo[0][("x[2]", "x[2]")] == -2.0
    # quadratic
    assert bqm_qubo[0][("x[1]", "x[2]")] == -3.0
    # offset
    assert bqm_qubo[1] == 0.0


def test_convert_to_polynomial(model_ising):
    assert model_ising._label_to_index["x[1]"] == 0
    assert model_ising._label_to_index["x[2]"] == 1
    assert model_ising._index_to_label[0] == "x[1]"
    assert model_ising._index_to_label[1] == "x[2]"

    polynomial = model_ising.to_polynomial()
    assert [0, 0, -1.0] in polynomial
    assert [1, 1, -2.0] in polynomial
    assert [0, 1, -3.0] in polynomial
    assert len(polynomial) == 3


################################
# Built-in functions
################################


def test_physical_model_repr(simple_model):
    assert isinstance(simple_model.__repr__(), str)
    assert "PhysicalModel({" in simple_model.__repr__()
    assert "'mtype':" in simple_model.__repr__()
    assert "'raw_interactions':" in simple_model.__repr__()


def test_physical_model_str(simple_model):
    assert isinstance(simple_model.__str__(), str)
    assert "PHYSICAL MODEL" in simple_model.__str__()
    assert "mtype:" in simple_model.__str__()
    assert "raw_interactions:" in simple_model.__str__()
    assert "linear:" in simple_model.__str__()
    assert "quadratic:" in simple_model.__str__()
