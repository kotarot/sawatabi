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

import pyqubo
import pytest

from sawatabi.model import LogicalModel


@pytest.fixture
def model():
    return LogicalModel(type="ising")


################################
# Model
################################


@pytest.mark.parametrize("type", ["ising", "qubo"])
def test_logical_model_constructor(type):
    model = LogicalModel(type=type)
    assert model.get_type() == type
    assert model._type == type


def test_logical_model_invalid_type(model):
    with pytest.raises(ValueError):
        LogicalModel()

    with pytest.raises(ValueError):
        LogicalModel(type="invalidtype")

    with pytest.raises(ValueError):
        LogicalModel(type=12345)


################################
# Variables
################################


@pytest.mark.parametrize("shape", [(2,), (3, 4), (5, 6, 7)])
def test_logical_model_variables(shape):
    model = LogicalModel(type="ising")
    x = model.variables("x", shape=shape)

    assert len(model.get_variables()) == 1
    assert "x" in model.get_variables()
    assert x.shape == shape
    assert isinstance(x, pyqubo.Array)
    assert model.get_variables_by_name("x") == x
    assert id(model.get_variables_by_name("x")) == id(x)
    assert id(model.get_variables()["x"]) == id(x)


@pytest.mark.parametrize("shape", [(2,), (3, 4), (5, 6, 7)])
def test_logical_model_multi_variables(shape):
    model = LogicalModel(type="qubo")
    x = model.variables("x", shape=shape)
    assert len(model.get_variables()) == 1
    assert "x" in model.get_variables()
    assert "y" not in model.get_variables()
    with pytest.raises(KeyError):
        model.get_variables_by_name("y")

    y = model.variables("y", shape=shape)
    assert len(model.get_variables()) == 2
    assert "x" in model.get_variables()
    assert "y" in model.get_variables()
    assert y.shape == shape
    assert isinstance(y, pyqubo.Array)
    assert model.get_variables_by_name("y") == y
    assert id(model.get_variables()["y"]) == id(y)


@pytest.mark.parametrize(
    "name,shape",
    [
        (12345, (2, 3)),
        ("x", 12345),
        ("x", ()),
        ("x", ("a", "b")),
    ],
)
def test_logical_model_variables_invalid(name, shape):
    model = LogicalModel(type="ising")
    with pytest.raises(TypeError):
        model.variables(name, shape=shape)


@pytest.mark.parametrize("vartype,modeltype", [("SPIN", "ising"), ("BINARY", "qubo")])
def test_logical_model_variables_from_pyqubo(vartype, modeltype):
    x = pyqubo.Array.create("x", shape=(2, 3), vartype=vartype)
    model = LogicalModel(type=modeltype)
    x_when_applied = model.variables(x)
    x_from_model = model.get_variables_by_name("x")
    assert id(x) == id(x_when_applied)
    assert x == x_when_applied
    assert id(x) == id(x_from_model)
    assert x == x_from_model


@pytest.mark.parametrize("vartype,modeltype", [("SPIN", "qubo"), ("BINARY", "ising")])
def test_logical_model_variables_from_pyqubo_mismatch(vartype, modeltype):
    x = pyqubo.Array.create("x", shape=(2, 3), vartype=vartype)
    model = LogicalModel(type=modeltype)
    with pytest.raises(TypeError):
        model.variables(x)


################################
# Select
################################


def test_logical_model_select(model):
    with pytest.raises(NotImplementedError):
        model.select_variable()

    with pytest.raises(NotImplementedError):
        model.select_interaction()


################################
# Add
################################


def test_logical_model_add(model):
    x = model.variables("x", shape=(2, 2))

    model.add_interaction(x[0, 0], coefficient=1.0)
    # TODO: Check result
    model.add_interaction(x[0, 1], coefficient=2.0, scale=0.1)
    # TODO: Check result
    model.add_interaction(x[1, 0], coefficient=3.0, attributes={"foo": "bar"})
    # TODO: Check result
    model.add_interaction((x[0, 0], x[1, 1]), coefficient=-4.0, timestamp=1234567890123)
    # TODO: Check result


def test_logical_model_add_invalid(model):
    x = model.variables("x", shape=(3,))

    with pytest.raises(ValueError):
        model.add_interaction(target=None)

    with pytest.raises(TypeError):
        model.add_interaction()

    with pytest.raises(TypeError):
        model.add_interaction("invalid type", coefficient=1.0)

    with pytest.raises(TypeError):
        model.add_interaction((x[0], x[1], x[2]), coefficient=1.0)

    with pytest.raises(TypeError):
        model.add_interaction(("a", "b"), coefficient=1.0)

    with pytest.raises(TypeError):
        model.add_interaction(x[0], coefficient="invalid type")

    with pytest.raises(TypeError):
        model.add_interaction(x[0], scale="invalid type")

    with pytest.raises(TypeError):
        model.add_interaction(x[0], attributes="invalid type")

    with pytest.raises(TypeError):
        model.add_interaction(x[0], timestamp="invalid type")


################################
# Update
################################


def test_logical_model_update(model):
    x = model.variables("x", shape=(1,))

    model.add_interaction(x[0], coefficient=1.0)
    # TODO: Check result
    model.update_interaction(x[0], coefficient=10.0)
    # TODO: Check result
    model.update_interaction(target=x[0], coefficient=100.0)
    # TODO: Check result
    model.update_interaction(name="x[0]", coefficient=1000.0)
    # TODO: Check result


def test_logical_model_update_invalid(model):
    x = model.variables("x", shape=(3,))
    model.add_interaction(x[0], coefficient=1.0)

    with pytest.raises(ValueError):
        model.update_interaction()

    with pytest.raises(ValueError):
        model.update_interaction(x[0], name="x[0]")

    with pytest.raises(KeyError):
        model.update_interaction(x[1], coefficient=1.0)

    with pytest.raises(TypeError):
        model.update_interaction("invalid type", coefficient=1.0)

    with pytest.raises(TypeError):
        model.update_interaction((x[0], x[1], x[2]), coefficient=1.0)

    with pytest.raises(TypeError):
        model.update_interaction(("a", "b"), coefficient=1.0)

    with pytest.raises(TypeError):
        model.add_interaction(x[0], coefficient="invalid type")

    with pytest.raises(TypeError):
        model.add_interaction(x[0], scale="invalid type")

    with pytest.raises(TypeError):
        model.add_interaction(x[0], attributes="invalid type")

    with pytest.raises(TypeError):
        model.add_interaction(x[0], timestamp="invalid type")


################################
# Remove
################################


def test_logical_model_remove(model):
    with pytest.raises(NotImplementedError):
        model.remove_interaction()


################################
# Erase
################################


def test_logical_model_erase(model):
    with pytest.raises(NotImplementedError):
        model.erase_variable()


################################
# Fix
################################


def test_logical_model_fix(model):
    with pytest.raises(NotImplementedError):
        model.fix_variable()


################################
# PyQUBO
################################


def test_logical_model_pyqubo(model):
    x, y = pyqubo.Spin("x"), pyqubo.Spin("y")
    exp = 2 * x * y + pyqubo.Placeholder("a") * x

    with pytest.raises(NotImplementedError):
        model.from_pyqubo(exp)


def test_logical_model_pyqubo_invalid(model):
    with pytest.raises(TypeError):
        model.from_pyqubo("invalid type")


################################
# Constraints
################################


def test_logical_model_n_hot_constraint_1(model):
    x = model.variables("x", shape=(3,))

    model.n_hot_constraint(x[0], n=1)
    assert len(model.get_constraints()) == 1
    assert "Default N-hot Constraint" in model.get_constraints()
    assert model.get_constraints_by_label("Default N-hot Constraint")._n == 1

    model.n_hot_constraint(x[(slice(1, 3),)], n=1)
    assert len(model.get_constraints()) == 1
    assert "Default N-hot Constraint" in model.get_constraints()
    assert model.get_constraints_by_label("Default N-hot Constraint")._n == 1


def test_logical_model_n_hot_constraint_2(model):
    y = model.variables("y", shape=(2, 2))

    model.n_hot_constraint(y[0, :], n=2, label="my label")
    assert len(model.get_constraints()) == 1
    assert "my label" in model.get_constraints()
    assert model.get_constraints_by_label("my label")._n == 2

    model.n_hot_constraint(y[1, :], n=2, label="my label")
    assert len(model.get_constraints()) == 1
    assert "my label" in model.get_constraints()
    assert model.get_constraints_by_label("my label")._n == 2


def test_logical_model_multi_n_hot_constraints(model):
    x = model.variables("x", shape=(2, 4))

    model.n_hot_constraint(x[0, :], n=1, label="l1")
    assert len(model.get_constraints()) == 1
    assert "l1" in model.get_constraints()
    assert "l2" not in model.get_constraints()
    assert model.get_constraints_by_label("l1")._n == 1
    with pytest.raises(KeyError):
        model.get_constraints_by_label("l2")

    model.n_hot_constraint(x[1, :], n=1, label="l2")
    assert len(model.get_constraints()) == 2
    assert "l1" in model.get_constraints()
    assert "l2" in model.get_constraints()
    assert model.get_constraints_by_label("l1")._n == 1
    assert model.get_constraints_by_label("l2")._n == 1


def test_logical_model_n_hot_constraint_valueerror(model):
    x = model.variables("x", shape=(10,))
    with pytest.raises(ValueError):
        model.n_hot_constraint(x[0], n=-10)
    with pytest.raises(ValueError):
        model.n_hot_constraint(x[0], n=0)


def test_logical_model_n_hot_constraint_typeerror(model):
    z = model.variables("z", shape=(4, 4, 4))

    with pytest.raises(TypeError):
        model.n_hot_constraint("invalid type")

    with pytest.raises(TypeError):
        model.n_hot_constraint(z[0, 0, :], n="invalid type")

    with pytest.raises(TypeError):
        model.n_hot_constraint(z[0, 0, :], scale="invalid type")

    with pytest.raises(TypeError):
        model.n_hot_constraint(z[0, 0, :], label=12345)


def test_logical_model_dependency_constraint(model):
    x = model.variables("x", shape=(2, 3))

    with pytest.raises(NotImplementedError):
        model.dependency_constraint(x[0, :], x[1, :])


################################
# Utils
################################


def test_logical_model_utils(model):
    other_model = LogicalModel(type="ising")
    with pytest.raises(NotImplementedError):
        model.merge(other_model)

    placeholder = {"a": 10.0}
    with pytest.raises(NotImplementedError):
        model.convert_to_physical(placeholder=placeholder)

    with pytest.raises(NotImplementedError):
        model.convert_type()
