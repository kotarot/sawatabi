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


def test_logical_model_invalid_type(model):
    with pytest.raises(ValueError):
        model = LogicalModel()  # noqa: F841

    with pytest.raises(ValueError):
        model = LogicalModel(type="anothertype")  # noqa: F841

    with pytest.raises(ValueError):
        model = LogicalModel(type=12345)  # noqa: F841


################################
# Variables
################################


@pytest.mark.parametrize("shape", [(2,), (3, 4), (5, 6, 7)])
def test_logical_model_variables(shape):
    model = LogicalModel(type="ising")
    x = model.variables("x", shape=shape)

    assert x.shape == shape


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
    import pyqubo

    x = pyqubo.Array.create("x", shape=(2, 3), vartype=vartype)
    model = LogicalModel(type=modeltype)
    model.variables(x)

    x_from_model = model.get_variables("FIXME")
    assert x == x_from_model


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

    model.add_interaction(x[0][0], coefficient=1.0)
    # TODO: Check result
    model.add_interaction(x[0][1], coefficient=2.0, scale=0.1)
    # TODO: Check result
    model.add_interaction(x[1][0], coefficient=3.0, attributes={"foo": "bar"})
    # TODO: Check result
    model.add_interaction((x[0][0], x[1][1]), coefficient=-4.0, timestamp=1234567890123)
    # TODO: Check result


def test_logical_model_add_invalid(model):
    x = model.variables("x", shape=(3,))

    with pytest.raises(ValueError):
        model.add_interaction(target=None)

    with pytest.raises(TypeError):
        model.add_interaction()

    with pytest.raises(TypeError):
        model.add_interaction("another type", coefficient=1.0)

    with pytest.raises(TypeError):
        model.add_interaction((x[0], x[1], x[2]), coefficient=1.0)

    with pytest.raises(TypeError):
        model.add_interaction(("a", "b"), coefficient=1.0)


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
        model.update_interaction("another type", coefficient=1.0)

    with pytest.raises(TypeError):
        model.update_interaction((x[0], x[1], x[2]), coefficient=1.0)

    with pytest.raises(TypeError):
        model.update_interaction(("a", "b"), coefficient=1.0)


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
        model.from_pyqubo("another type")


################################
# Constraints
################################


def test_logical_model_constraints(model):
    with pytest.raises(NotImplementedError):
        model.n_hot_constraint()

    with pytest.raises(NotImplementedError):
        model.dependency_constraint()
