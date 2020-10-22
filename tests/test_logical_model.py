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

import sawatabi.constants as constants
from sawatabi.model import LogicalModel


@pytest.fixture
def model():
    return LogicalModel(mtype="ising")


@pytest.fixture
def model_qubo():
    return LogicalModel(mtype="qubo")


################################
# Logical Model
################################


@pytest.mark.parametrize("mtype", ["ising", "qubo"])
def test_logical_model_constructor(mtype):
    model = LogicalModel(mtype=mtype)
    assert model.get_mtype() == mtype
    assert model._mtype == mtype


def test_logical_model_invalid_mtype(model):
    with pytest.raises(ValueError):
        LogicalModel()

    with pytest.raises(ValueError):
        LogicalModel(mtype="invalidtype")

    with pytest.raises(ValueError):
        LogicalModel(mtype=12345)


################################
# Variables
################################


@pytest.mark.parametrize("shape", [(2,), (3, 4), (5, 6, 7)])
def test_logical_model_variables(shape):
    model = LogicalModel(mtype="ising")
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
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape)
    assert len(model.get_variables()) == 1
    assert "x" in model.get_variables()
    assert "y" not in model.get_variables()
    assert model.get_variables_by_name("x") == x
    assert id(model.get_variables_by_name("x")) == id(x)
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
    model = LogicalModel(mtype="ising")
    with pytest.raises(TypeError):
        model.variables(name, shape=shape)


def test_logical_model_variables_comma(model):
    # We cannot name variables whose name contains a comma
    with pytest.raises(AssertionError):
        model.variables("x*y", shape=(2, 2))


@pytest.mark.parametrize("vartype,mtype", [("SPIN", "ising"), ("BINARY", "qubo")])
def test_logical_model_variables_from_pyqubo(vartype, mtype):
    x = pyqubo.Array.create("x", shape=(2, 3), vartype=vartype)
    model = LogicalModel(mtype=mtype)
    x_when_applied = model.variables(x)
    x_from_model = model.get_variables_by_name("x")
    assert id(x) == id(x_when_applied)
    assert x == x_when_applied
    assert id(x) == id(x_from_model)
    assert x == x_from_model


@pytest.mark.parametrize("vartype,mtype", [("SPIN", "qubo"), ("BINARY", "ising")])
def test_logical_model_variables_from_pyqubo_mismatch(vartype, mtype):
    x = pyqubo.Array.create("x", shape=(2, 3), vartype=vartype)
    model = LogicalModel(mtype=mtype)
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

    res = model.add_interaction(x[0, 0], coefficient=1.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 0
    assert res["name"] == "x[0][0]"
    assert res["key"] == "x[0][0]"
    assert res["interacts"] == x[0, 0]
    assert id(res["interacts"]) == id(x[0, 0])
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0][0]"]["coefficient"] == 1.0
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0][0]"]["scale"] == 1.0

    res = model.add_interaction(x[0, 1], coefficient=2.0, scale=0.1)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 2
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 0
    assert res["name"] == "x[0][1]"
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0][1]"]["coefficient"] == 2.0
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0][1]"]["scale"] == 0.1

    res = model.add_interaction(x[1, 0], coefficient=3.0, attributes={"foo": "bar"})
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 3
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 0
    assert res["name"] == "x[1][0]"
    assert model._interactions[constants.INTERACTION_LINEAR]["x[1][0]"]["coefficient"] == 3.0
    assert model._interactions[constants.INTERACTION_LINEAR]["x[1][0]"]["attributes"]["foo"] == "bar"

    res = model.add_interaction((x[0, 0], x[0, 1]), coefficient=-4.0, timestamp=1234567890123)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 3
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert res["name"] == "x[0][0]*x[0][1]"
    assert res["key"] == ("x[0][0]", "x[0][1]")
    assert res["interacts"] == (x[0, 0], x[0, 1])
    assert id(res["interacts"][0]) == id(x[0, 0])
    assert id(res["interacts"][1]) == id(x[0, 1])
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[0][0]*x[0][1]"]["coefficient"] == -4.0
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[0][0]*x[0][1]"]["timestamp"] == 1234567890123

    # Check key order
    res = model.add_interaction((x[1, 1], x[1, 0]), coefficient=-4.0, timestamp=1234567890123)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 3
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 2
    assert res["name"] == "x[1][0]*x[1][1]"
    assert res["key"] == ("x[1][0]", "x[1][1]")
    assert res["interacts"] == (x[1, 0], x[1, 1])
    assert id(res["interacts"][0]) == id(x[1, 0])
    assert id(res["interacts"][1]) == id(x[1, 1])
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[1][0]*x[1][1]"]["coefficient"] == -4.0
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[1][0]*x[1][1]"]["timestamp"] == 1234567890123


def test_logical_model_add_invalid_arguments(model):
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


def test_logical_model_add_duplicate(model):
    x = model.variables("x", shape=(2,))

    res = model.add_interaction(x[0], coefficient=1.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 0
    assert res["name"] == "x[0]"

    res = model.add_interaction(x[0], name="my name", coefficient=1.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 2
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 0
    assert res["name"] == "my name"


def test_logical_model_add_duplicate_invalid(model):
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], name="my name", coefficient=1.0)

    with pytest.raises(ValueError):
        model.add_interaction(x[0], coefficient=2.0)

    with pytest.raises(ValueError):
        model.add_interaction(x[1], name="my name", coefficient=2.0)


################################
# Update
################################


def test_logical_model_update(model):
    x = model.variables("x", shape=(2,))

    # initialize
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction((x[0], x[1]), coefficient=2.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["coefficient"] == 1.0
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[0]*x[1]"]["coefficient"] == 2.0

    # update by a variable
    res = model.update_interaction(x[0], coefficient=10.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["coefficient"] == 10.0
    assert res["name"] == "x[0]"
    assert res["key"] == "x[0]"
    assert res["interacts"] == x[0]
    assert id(res["interacts"]) == id(x[0])

    # update by a target
    model.update_interaction(target=x[0], coefficient=100.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["coefficient"] == 100.0

    # update by a name
    model.update_interaction(name="x[0]", coefficient=1000.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["coefficient"] == 1000.0

    # update by a pair of variables
    res = model.update_interaction(target=(x[0], x[1]), coefficient=20.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[0]*x[1]"]["coefficient"] == 20.0
    assert res["name"] == "x[0]*x[1]"
    assert res["key"] == ("x[0]", "x[1]")
    assert res["interacts"] == (x[0], x[1])
    assert id(res["interacts"][0]) == id(x[0])
    assert id(res["interacts"][1]) == id(x[1])

    # update by a pair of variables (reversed order)
    res = model.update_interaction(target=(x[1], x[0]), coefficient=200.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[0]*x[1]"]["coefficient"] == 200.0
    assert res["name"] == "x[0]*x[1]"
    assert res["key"] == ("x[0]", "x[1]")
    assert res["interacts"] == (x[0], x[1])
    assert id(res["interacts"][0]) == id(x[0])
    assert id(res["interacts"][1]) == id(x[1])

    # update by a name
    model.update_interaction(name="x[0]*x[1]", coefficient=2000.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[0]*x[1]"]["coefficient"] == 2000.0


def test_logical_model_update_by_custom_names(model):
    y = model.variables("y", shape=(2,))
    n1 = "custom interaction 1"
    n2 = "custom interaction 2"
    n3 = "custom interaction 3"

    # initialize
    model.add_interaction(y[0], coefficient=-1.0, name=n1)
    model.add_interaction((y[0], y[1]), coefficient=-2.0, name=n2)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert model._interactions[constants.INTERACTION_LINEAR][n1]["coefficient"] == -1.0
    assert model._interactions[constants.INTERACTION_QUADRATIC][n2]["coefficient"] == -2.0

    model.update_interaction(name=n1, coefficient=-10.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert model._interactions[constants.INTERACTION_LINEAR][n1]["coefficient"] == -10.0

    model.update_interaction(name=n2, coefficient=-20.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(model._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert model._interactions[constants.INTERACTION_QUADRATIC][n2]["coefficient"] == -20.0

    with pytest.raises(KeyError):
        model.update_interaction(name=n3, coefficient=-30.0)


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


def test_logical_model_remove_by_target(model):
    x = model.variables("x", shape=(2,))

    # initialize
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=10.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 2

    # remove
    res = model.remove_interaction(x[0])
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 2
    assert res["name"] == "x[0]"
    assert res["key"] == "x[0]"
    assert res["coefficient"] == 1.0
    assert res["dirty"] == True
    assert res["removed"] == True

    res = model.remove_interaction(name="x[1]")
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 2
    assert res["name"] == "x[1]"
    assert res["key"] == "x[1]"
    assert res["coefficient"] == 10.0
    assert res["dirty"] == True
    assert res["removed"] == True


def test_logical_model_remove_by_name(model):
    x = model.variables("x", shape=(2,))

    # initialize
    model.add_interaction(x[0], "my name", coefficient=1.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1

    # remove
    res = model.remove_interaction(name="my name")
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 1
    assert res["removed"] == True


def test_logical_model_remove_invalid(model):
    x = model.variables("x", shape=(3,))
    model.add_interaction(x[0], coefficient=1.0)

    with pytest.raises(ValueError):
        model.remove_interaction()

    with pytest.raises(ValueError):
        model.remove_interaction(x[0], name="x[0]")

    with pytest.raises(KeyError):
        model.remove_interaction(x[1])

    with pytest.raises(TypeError):
        model.remove_interaction("invalid type")

    with pytest.raises(TypeError):
        model.update_interaction((x[0], x[1], x[2]))

    with pytest.raises(TypeError):
        model.update_interaction(("a", "b"))


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


def test_logical_model_n_hot_constraint_3(model):
    x = model.variables("x", shape=(3,))
    default_label = "Default N-hot Constraint"

    model.n_hot_constraint(x, n=2)
    assert len(model.get_constraints()) == 1
    assert default_label in model.get_constraints()
    assert model.get_constraints_by_label(default_label)._n == 2
    assert len(model.get_constraints_by_label(default_label)._variables) == 3

    model.n_hot_constraint(x, n=2)  # double
    assert len(model.get_constraints()) == 1
    assert default_label in model.get_constraints()
    assert model.get_constraints_by_label(default_label)._n == 2
    assert len(model.get_constraints_by_label(default_label)._variables) == 3

    model.n_hot_constraint(x, n=2)  # partially
    assert len(model.get_constraints()) == 1
    assert default_label in model.get_constraints()
    assert model.get_constraints_by_label(default_label)._n == 2
    assert len(model.get_constraints_by_label(default_label)._variables) == 3


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
    a = pyqubo.Spin("a")

    with pytest.raises(TypeError):
        model.n_hot_constraint("invalid type")

    # TODO: This error should be raises, but not implemented yet.
    # with pytest.raises(ValueError):
    #     model.n_hot_constraint(a)

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


@pytest.mark.parametrize("mtype", ["ising", "qubo"])
def test_logical_model_convert(mtype):
    model = LogicalModel(mtype=mtype)
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction((x[0], x[1]), coefficient=-1.0)
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["dirty"] == True
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[0]*x[1]"]["dirty"] == True

    physical = model.convert_to_physical()
    assert physical.get_mtype() == mtype
    assert len(physical._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(physical._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert physical._interactions[constants.INTERACTION_LINEAR]["x[0]"] == 1.0
    assert physical._interactions[constants.INTERACTION_QUADRATIC][("x[0]", "x[1]")] == -1.0
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["dirty"] == False
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[0]*x[1]"]["dirty"] == False


def test_logical_model_convert_with_doubled_interactions(model):
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=2.0, scale=0.5)
    model.add_interaction(x[0], name="additional x[0]", coefficient=-2.0)
    model.add_interaction((x[0], x[1]), coefficient=-2.0, scale=0.5)
    model.add_interaction((x[0], x[1]), name="additional x[0]*x[1]", coefficient=2.0)
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["dirty"] == True
    assert model._interactions[constants.INTERACTION_LINEAR]["additional x[0]"]["dirty"] == True
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[0]*x[1]"]["dirty"] == True
    assert model._interactions[constants.INTERACTION_QUADRATIC]["additional x[0]*x[1]"]["dirty"] == True

    physical = model.convert_to_physical()
    assert len(physical._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(physical._interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert physical._interactions[constants.INTERACTION_LINEAR]["x[0]"] == -1.0
    assert physical._interactions[constants.INTERACTION_QUADRATIC][("x[0]", "x[1]")] == 1.0
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["dirty"] == False
    assert model._interactions[constants.INTERACTION_LINEAR]["additional x[0]"]["dirty"] == False
    assert model._interactions[constants.INTERACTION_QUADRATIC]["x[0]*x[1]"]["dirty"] == False
    assert model._interactions[constants.INTERACTION_QUADRATIC]["additional x[0]*x[1]"]["dirty"] == False


def test_logical_model_convert_with_remove(model):
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=1.0)
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 2
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["dirty"] == True
    assert model._interactions[constants.INTERACTION_LINEAR]["x[1]"]["dirty"] == True
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["removed"] == False
    assert model._interactions[constants.INTERACTION_LINEAR]["x[1]"]["removed"] == False

    model.remove_interaction(x[1])
    assert len(model._interactions[constants.INTERACTION_LINEAR]) == 2
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["removed"] == False
    assert model._interactions[constants.INTERACTION_LINEAR]["x[1]"]["removed"] == True

    physical = model.convert_to_physical()
    assert len(physical._interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(physical._interactions[constants.INTERACTION_QUADRATIC]) == 0

    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["removed"] == False
    assert "x[1]" not in model._interactions[constants.INTERACTION_LINEAR]
    assert model._interactions[constants.INTERACTION_LINEAR]["x[0]"]["dirty"] == False


@pytest.mark.parametrize("n,s", [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 100), (10, 400)])
def test_logical_model_convert_with_n_hot_constraint_qubo(n, s):
    # n out of s variables should be 1
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(s,))
    model.n_hot_constraint(x, n=n)
    physical = model.convert_to_physical()

    for i in range(s):
        assert physical._interactions[constants.INTERACTION_LINEAR]["x[{}]".format(i)] == 2 * n - 1.0
    for i in range(s):
        for j in range(s):
            l1 = "x[{}]".format(i)
            l2 = "x[{}]".format(j)
            if l1 < l2:
                assert physical._interactions[constants.INTERACTION_QUADRATIC][(l1, l2)] == -2.0


@pytest.mark.parametrize("n,s", [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 100), (10, 400)])
def test_logical_model_convert_with_n_hot_constraint_ising(n, s):
    # n out of s spins should be +1
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(s,))
    model.n_hot_constraint(x, n=n)
    physical = model.convert_to_physical()

    for i in range(s):
        assert physical._interactions[constants.INTERACTION_LINEAR]["x[{}]".format(i)] == -1.0 * (s - 2 * n)
    for i in range(s):
        for j in range(s):
            l1 = "x[{}]".format(i)
            l2 = "x[{}]".format(j)
            if l1 < l2:
                assert physical._interactions[constants.INTERACTION_QUADRATIC][(l1, l2)] == -1.0


def test_logical_model_convert_with_n_hot_constraint_randomly_qubo(model_qubo):
    x = model_qubo.variables("x", shape=(4,))
    model_qubo.n_hot_constraint(x[(slice(0, 2),)], n=1, strength=10)
    model_qubo.n_hot_constraint(x[1], n=1, strength=10)
    model_qubo.n_hot_constraint(x[2], n=1, strength=10)
    model_qubo.n_hot_constraint(x, n=1, strength=10)
    physical = model_qubo.convert_to_physical()

    for i in range(3):
        assert physical._interactions[constants.INTERACTION_LINEAR]["x[{}]".format(i)] == 10.0
    for i in range(2):
        for j in range(i + 1, 3):
            assert physical._interactions[constants.INTERACTION_QUADRATIC][("x[{}]".format(i), "x[{}]".format(j))] == -20.0


def test_logical_model_convert_with_n_hot_constraint_randomly_ising(model):
    x = model.variables("x", shape=(4,))
    model.n_hot_constraint(x[(slice(0, 2),)], n=1)
    physical = model.convert_to_physical()
    model.n_hot_constraint(x[1], n=1)
    physical = model.convert_to_physical()
    model.n_hot_constraint(x[2], n=1)
    physical = model.convert_to_physical()
    model.n_hot_constraint(x, n=1)
    physical = model.convert_to_physical()

    for i in range(3):
        assert physical._interactions[constants.INTERACTION_LINEAR]["x[{}]".format(i)] == -2.0
    for i in range(2):
        for j in range(i + 1, 3):
            assert physical._interactions[constants.INTERACTION_QUADRATIC][("x[{}]".format(i), "x[{}]".format(j))] == -1.0


def test_logical_model_convert_with_placeholder(model):
    placeholder = {"a": 10.0}
    model.convert_to_physical(placeholder=placeholder)
    # TODO


def test_logical_model_utils(model):
    other_model = LogicalModel(mtype="ising")
    with pytest.raises(NotImplementedError):
        model.merge(other_model)

    with pytest.raises(NotImplementedError):
        model.convert_mtype()


################################
# Built-in functions
################################


def test_logical_model_repr(model):
    assert isinstance(model.__repr__(), str)
    assert "LogicalModel({" in model.__repr__()
    assert "'mtype':" in model.__repr__()
    assert "'variables':" in model.__repr__()
    assert "'interactions':" in model.__repr__()
    assert "'constraints':" in model.__repr__()


def test_logical_model_str(model):
    assert isinstance(model.__str__(), str)
    assert "LOGICAL MODEL" in model.__str__()
    assert "mtype:" in model.__str__()
    assert "variables:" in model.__str__()
    assert "interactions:" in model.__str__()
    assert "linear:" in model.__str__()
    assert "quadratic:" in model.__str__()
    assert "constraints:" in model.__str__()
