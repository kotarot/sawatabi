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

################################
# Model
################################


@pytest.mark.parametrize("type", ["ising", "qubo"])
def test_logical_model_constructor(type):
    model = LogicalModel(type=type)
    assert model.get_type() == type


def test_logical_model_invalid_type():
    with pytest.raises(ValueError):
        model = LogicalModel()  # noqa: F841

    with pytest.raises(ValueError):
        model = LogicalModel(type="othertype")  # noqa: F841

    with pytest.raises(ValueError):
        model = LogicalModel(type=12345)  # noqa: F841


################################
# Array
################################


@pytest.mark.parametrize("shape", [(2,), (3, 4), (5, 6, 7)])
def test_logical_model_array(shape):
    model = LogicalModel(type="ising")
    x = model.array("x", shape=shape)
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
def test_logical_model_invalid_arrays(name, shape):
    model = LogicalModel(type="ising")
    with pytest.raises(TypeError):
        model.array(name, shape=shape)


def test_logical_model_arrays_from_pyqubo():
    import pyqubo

    x = pyqubo.Array.create("x", shape=(2, 3), vartype="BINARY")
    model = LogicalModel(type="qubo")
    model.array(x)

    x_from_model = model.get_array()
    assert x == x_from_model


################################
# Add
################################


def test_logical_model_add():
    model = LogicalModel(type="ising")

    with pytest.raises(NotImplementedError):
        model.add_variable()

    with pytest.raises(NotImplementedError):
        model.add_variables()

    with pytest.raises(NotImplementedError):
        model.add_interaction()

    with pytest.raises(NotImplementedError):
        model.add_interactions()


################################
# Remove
################################


def test_logical_model_remove():
    model = LogicalModel(type="ising")

    with pytest.raises(NotImplementedError):
        model.remove_variable()

    with pytest.raises(NotImplementedError):
        model.remove_interaction()


################################
# Fix
################################


def test_logical_model_fix():
    model = LogicalModel(type="ising")

    with pytest.raises(NotImplementedError):
        model.fix_variable()

    with pytest.raises(NotImplementedError):
        model.fix_interaction()


################################
# PyQUBO
################################


def test_logical_model_pyqubo():
    model = LogicalModel(type="ising")
    x, y = pyqubo.Spin("x"), pyqubo.Spin("y")
    exp = 2 * x * y + pyqubo.Placeholder("a") * x

    with pytest.raises(TypeError):
        model.from_pyqubo("another type")

    with pytest.raises(NotImplementedError):
        model.from_pyqubo(exp)
