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
import pyqubo
import pytest

import sawatabi.constants as constants
from sawatabi.model import LogicalModel


@pytest.fixture
def ising():
    return LogicalModel(mtype="ising")


@pytest.fixture
def qubo():
    return LogicalModel(mtype="qubo")


################################
# Logical Model
################################


@pytest.mark.parametrize("mtype", ["ising", "qubo"])
def test_logical_model_constructor(mtype):
    model = LogicalModel(mtype=mtype)
    assert model.get_mtype() == mtype
    assert model._mtype == mtype


def test_logical_model_invalid_mtype():
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


def test_logical_model_variables_comma(ising):
    # We cannot name variables whose name contains a comma
    with pytest.raises(AssertionError):
        ising.variables("x*y", shape=(2, 2))


@pytest.mark.parametrize("initial_shape,additional_shape,expected_shape", [((2,), (1,), (3,)), ((44, 33), (22, 11), (66, 44))])
def test_logical_model_variables_append(initial_shape, additional_shape, expected_shape):
    model = LogicalModel(mtype="ising")
    model.variables("x", shape=initial_shape)
    assert "x" in model.get_variables()
    assert model.get_variables_by_name("x").shape == initial_shape

    model.append("x", shape=additional_shape)
    assert model.get_variables_by_name("x").shape == expected_shape


@pytest.mark.parametrize("shape", [(2,), (33, 44)])
def test_logical_model_variables_append_without_initialize(shape):
    model = LogicalModel(mtype="ising")
    # The following operation will be successful with a UserWarning.
    with pytest.warns(UserWarning):
        model.append("x", shape=shape)
    assert "x" in model.get_variables()
    assert model.get_variables_by_name("x").shape == shape


def test_logical_model_variables_append_invalid(ising):
    ising.variables("x", shape=(2, 2))

    with pytest.raises(TypeError):
        ising.append("x", shape=("a", "b"))


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
# PyQUBO
################################


def test_logical_model_from_pyqubo_expression(qubo):
    x = qubo.variables("x", shape=(10,))
    y = qubo.variables("y", shape=(10,))

    sum_x = sum(x[i] for i in range(10))
    sum_y = sum(y[i] for i in range(10))
    hamiltonian = (sum_x - sum_y) ** 2

    qubo.from_pyqubo(hamiltonian)

    qubo._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug

    for i in range(10):
        assert qubo._interactions[qubo._interactions["name"] == f"x[{i}]"]["coefficient"].values[0] == 1.0
    for i in range(9):
        for j in range(i + 1, 10):
            assert qubo._interactions[qubo._interactions["name"] == f"x[{i}]*x[{j}]"]["coefficient"].values[0] == 2.0
            assert qubo._interactions[qubo._interactions["name"] == f"y[{i}]*y[{j}]"]["coefficient"].values[0] == 2.0
            assert qubo._interactions[qubo._interactions["name"] == f"x[{i}]*y[{j}]"]["coefficient"].values[0] == -2.0

    assert qubo._offset == 0.0


def test_logical_model_from_pyqubo_model_with_placeholder(qubo):
    x = qubo.variables("x", shape=(10, 2))
    y = qubo.variables("y", shape=(10, 2))

    sum_x = sum(x[i, 0] for i in range(10))
    sum_y = sum(y[i, 0] for i in range(10))
    hamiltonian = pyqubo.Placeholder("A") * (sum_x - sum_y) ** 2 + 10.0
    pyqubo_model = hamiltonian.compile()

    qubo.from_pyqubo(pyqubo_model)

    # We cannot evaluate cofficient values before placeholders are resolved,
    # so convert it to a physical model.
    physical = qubo.to_physical({"A": 2.0})

    for i in range(10):
        assert physical._raw_interactions[constants.INTERACTION_LINEAR][f"x[{i}][0]"] == 2.0
    for i in range(10):
        for j in range(i + 1, 10):
            assert physical._raw_interactions[constants.INTERACTION_QUADRATIC][(f"x[{i}][0]", f"x[{j}][0]")] == 4.0
            assert physical._raw_interactions[constants.INTERACTION_QUADRATIC][(f"y[{i}][0]", f"y[{j}][0]")] == 4.0
            assert physical._raw_interactions[constants.INTERACTION_QUADRATIC][(f"x[{i}][0]", f"y[{j}][0]")] == -4.0

    assert physical._offset == 10.0


def test_logical_model_from_pyqubo_invalid(qubo):
    with pytest.raises(TypeError):
        qubo.from_pyqubo("invalid type")


################################
# Constraints
################################


def test_logical_model_n_hot_constraint_1(ising):
    x = ising.variables("x", shape=(3,))

    ising.n_hot_constraint(x[0], n=1)
    assert len(ising.get_constraints()) == 1
    assert "Default N-hot Constraint" in ising.get_constraints()
    assert ising.get_constraints_by_label("Default N-hot Constraint")._n == 1

    ising.n_hot_constraint(target=x[(slice(1, 3),)], n=1)
    assert len(ising.get_constraints()) == 1
    assert "Default N-hot Constraint" in ising.get_constraints()
    assert ising.get_constraints_by_label("Default N-hot Constraint")._n == 1


def test_logical_model_n_hot_constraint_2(ising):
    y = ising.variables("y", shape=(2, 2))

    ising.n_hot_constraint(y[0, :], n=2, label="my label")
    assert len(ising.get_constraints()) == 1
    assert "my label" in ising.get_constraints()
    assert ising.get_constraints_by_label("my label")._n == 2

    # targte in list representation
    ising.n_hot_constraint(target=[y[1, 0], y[1, 1]], n=2, label="my label")
    assert len(ising.get_constraints()) == 1
    assert "my label" in ising.get_constraints()
    assert ising.get_constraints_by_label("my label")._n == 2


def test_logical_model_n_hot_constraint_3(ising):
    x = ising.variables("x", shape=(3,))
    default_label = "Default N-hot Constraint"

    ising.n_hot_constraint(x, n=2)
    assert len(ising.get_constraints()) == 1
    assert default_label in ising.get_constraints()
    assert ising.get_constraints_by_label(default_label)._n == 2
    assert len(ising.get_constraints_by_label(default_label)._variables) == 3

    ising.n_hot_constraint(x, n=2)  # double
    assert len(ising.get_constraints()) == 1
    assert default_label in ising.get_constraints()
    assert ising.get_constraints_by_label(default_label)._n == 2
    assert len(ising.get_constraints_by_label(default_label)._variables) == 3

    ising.n_hot_constraint(x, n=2)  # partially
    assert len(ising.get_constraints()) == 1
    assert default_label in ising.get_constraints()
    assert ising.get_constraints_by_label(default_label)._n == 2
    assert len(ising.get_constraints_by_label(default_label)._variables) == 3


def test_logical_model_multi_n_hot_constraints(ising):
    x = ising.variables("x", shape=(2, 4))

    ising.n_hot_constraint(x[0, :], n=1, label="l1")
    assert len(ising.get_constraints()) == 1
    assert "l1" in ising.get_constraints()
    assert "l2" not in ising.get_constraints()
    assert ising.get_constraints_by_label("l1")._n == 1
    with pytest.raises(KeyError):
        ising.get_constraints_by_label("l2")

    ising.n_hot_constraint(x[1, :], n=1, label="l2")
    assert len(ising.get_constraints()) == 2
    assert "l1" in ising.get_constraints()
    assert "l2" in ising.get_constraints()
    assert ising.get_constraints_by_label("l1")._n == 1
    assert ising.get_constraints_by_label("l2")._n == 1


def test_logical_model_n_hot_constraint_valueerror(ising):
    x = ising.variables("x", shape=(10,))

    with pytest.raises(ValueError):
        ising.n_hot_constraint(x[0], n=-10)

    with pytest.raises(ValueError):
        ising.n_hot_constraint(x[0], n=0)


def test_logical_model_n_hot_constraint_typeerror(ising):
    z = ising.variables("z", shape=(4, 4, 4))

    with pytest.raises(TypeError):
        ising.n_hot_constraint("invalid type")

    # TODO: This error should be raises, but not implemented yet.
    # a = pyqubo.Spin("a")
    # with pytest.raises(ValueError):
    #     ising.n_hot_constraint(a)

    with pytest.raises(TypeError):
        ising.n_hot_constraint(z[0, 0, :], n="invalid type")

    with pytest.raises(TypeError):
        ising.n_hot_constraint(z[0, 0, :], scale="invalid type")

    with pytest.raises(TypeError):
        ising.n_hot_constraint(z[0, 0, :], label=12345)

    with pytest.raises(TypeError):
        ising.n_hot_constraint(z[0, 0, :], label=12345)

    with pytest.raises(TypeError):
        ising.n_hot_constraint(["a", "b"], n=1)


def test_logical_model_dependency_constraint(ising):
    x = ising.variables("x", shape=(2, 3))

    with pytest.raises(NotImplementedError):
        ising.dependency_constraint(x[0, :], x[1, :])


################################
# Getters
################################


def test_logical_model_get_deleted_array(ising):
    x = ising.variables("x", shape=(2,))
    assert len(ising.get_deleted_array()) == 0
    ising.delete_variable(target=x[0])
    assert len(ising.get_deleted_array()) == 1
    assert "x[0]" in ising.get_deleted_array()


def test_logical_model_get_fixed_array(ising):
    x = ising.variables("x", shape=(2,))  # noqa: F841
    assert len(ising.get_fixed_array()) == 0
    # ising.fix_variable(target=x[0], value=1)
    # assert len(ising.get_fixed_array()) == 1


def test_logical_model_get_attributes(ising):
    x = ising.variables("x", shape=(2,))
    ising.add_interaction(x[0], coefficient=10.0)
    ising.add_interaction(x[1], coefficient=11.0, attributes={"foo": "bar"})

    attributes = ising.get_attributes(x[0])
    assert len(attributes) == 1
    assert np.isnan(attributes["attributes.foo"])

    attributes = ising.get_attributes(target=x[1])
    assert len(attributes) == 1
    assert attributes["attributes.foo"] == "bar"

    attributes = ising.get_attributes(name="x[1]")
    assert len(attributes) == 1
    assert attributes["attributes.foo"] == "bar"

    attribute = ising.get_attribute(x[1], key="attributes.foo")
    assert isinstance(attribute, str)
    assert attribute == "bar"

    with pytest.raises(KeyError):
        ising.get_attribute(name="x[1]", key="attributes.foofoo")


################################
# Built-in functions
################################


def test_logical_model_eq():
    model_a = _create_ising_model_for_eq()
    model_b = _create_ising_model_for_eq()
    assert model_a == model_b


def test_logical_model_ne():
    ising = LogicalModel(mtype="ising")
    qubo = LogicalModel(mtype="qubo")
    assert ising != qubo
    assert ising != "another type"


def _create_ising_model_for_eq():
    model = LogicalModel(mtype="ising")
    x = model.variables(name="x", shape=(4,))
    z = model.variables(name="z", shape=(4,))
    model.add_interaction(target=x[0], coefficient=1.1)
    model.add_interaction(target=(x[0], x[1]), coefficient=2.2, scale=3.3, attributes={"foo": "bar"})

    model.add_interaction(target=x[2], coefficient=4.4)
    model.add_interaction(target=x[3], coefficient=5.5)
    model.remove_interaction(target=x[2])
    model.fix_variable(target=x[3], value=1)

    model.n_hot_constraint(target=z, n=1)

    return model


def test_logical_model_repr(ising):
    ising.variables(name="x", shape=(10, 10))

    assert isinstance(ising.__repr__(), str)
    assert "LogicalModel({" in ising.__repr__()
    assert "'mtype':" in ising.__repr__()
    assert "'variables':" in ising.__repr__()
    assert "'x':" in ising.__repr__()
    assert "'interactions':" in ising.__repr__()
    assert "'offset':" in ising.__repr__()
    assert "'constraints':" in ising.__repr__()


def test_logical_model_str(ising):
    ising.variables(name="x", shape=(10, 10))

    assert isinstance(ising.__str__(), str)
    assert "LOGICAL MODEL" in ising.__str__()
    assert "mtype: ising" in ising.__str__()
    assert "variables: ['x']" in ising.__str__()
    assert "name: x" in ising.__str__()
    assert "interactions:" in ising.__str__()
    assert "offset:" in ising.__str__()
    assert "constraints:" in ising.__str__()
