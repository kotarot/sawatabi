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
# Converts
################################


@pytest.mark.parametrize("mtype", ["ising", "qubo"])
def test_logical_model_to_physical(mtype):
    model = LogicalModel(mtype=mtype)
    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction((x[0], x[1]), coefficient=-1.0)
    model._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug

    assert model._interactions[model._interactions["name"] == "x[0]"]["dirty"].values[0]
    assert model._interactions[model._interactions["name"] == "x[0]*x[1]"]["dirty"].values[0]

    physical = model.to_physical()
    model._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug

    assert physical.get_mtype() == mtype
    assert len(physical._raw_interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(physical._raw_interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert physical._raw_interactions[constants.INTERACTION_LINEAR]["x[0]"] == 1.0
    assert physical._raw_interactions[constants.INTERACTION_QUADRATIC][("x[0]", "x[1]")] == -1.0
    assert not model._interactions[model._interactions["name"] == "x[0]"]["dirty"].values[0]
    assert not model._interactions[model._interactions["name"] == "x[0]*x[1]"]["dirty"].values[0]

    assert physical._label_to_index["x[0]"] == 0
    assert physical._label_to_index["x[1]"] == 1
    assert len(physical._label_to_index) == 2
    assert physical._index_to_label[0] == "x[0]"
    assert physical._index_to_label[1] == "x[1]"
    assert len(physical._index_to_label) == 2


def test_logical_model_to_physical_with_doubled_interactions(ising):
    x = ising.variables("x", shape=(2,))
    ising.add_interaction(x[0], coefficient=2.0, scale=0.5)
    ising.add_interaction(x[0], name="additional x[0]", coefficient=-2.0)
    ising.add_interaction((x[0], x[1]), coefficient=-2.0, scale=0.5)
    ising.add_interaction((x[0], x[1]), name="additional x[0]*x[1]", coefficient=2.0)
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug

    assert ising._interactions[ising._interactions["name"] == "x[0]"]["dirty"].values[0]
    assert ising._interactions[ising._interactions["name"] == "additional x[0]"]["dirty"].values[0]
    assert ising._interactions[ising._interactions["name"] == "x[0]*x[1]"]["dirty"].values[0]
    assert ising._interactions[ising._interactions["name"] == "additional x[0]*x[1]"]["dirty"].values[0]

    physical = ising.to_physical()
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug

    assert len(physical._raw_interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(physical._raw_interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert physical._raw_interactions[constants.INTERACTION_LINEAR]["x[0]"] == -1.0
    assert physical._raw_interactions[constants.INTERACTION_QUADRATIC][("x[0]", "x[1]")] == 1.0
    assert not ising._interactions[ising._interactions["name"] == "x[0]"]["dirty"].values[0]
    assert not ising._interactions[ising._interactions["name"] == "additional x[0]"]["dirty"].values[0]
    assert not ising._interactions[ising._interactions["name"] == "x[0]*x[1]"]["dirty"].values[0]
    assert not ising._interactions[ising._interactions["name"] == "additional x[0]*x[1]"]["dirty"].values[0]


def test_logical_model_to_physical_with_remove(ising):
    x = ising.variables("x", shape=(2,))
    ising.add_interaction(x[0], coefficient=1.0)
    ising.add_interaction(x[1], coefficient=1.0)
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug

    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == "x[0]"]["dirty"].values[0]
    assert ising._interactions[ising._interactions["name"] == "x[1]"]["dirty"].values[0]
    assert not ising._interactions[ising._interactions["name"] == "x[0]"]["removed"].values[0]
    assert not ising._interactions[ising._interactions["name"] == "x[1]"]["removed"].values[0]

    ising.remove_interaction(x[1])
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug

    assert len(ising._interactions) == 2
    assert not ising._interactions[ising._interactions["name"] == "x[0]"]["removed"].values[0]
    assert ising._interactions[ising._interactions["name"] == "x[1]"]["removed"].values[0]

    physical = ising.to_physical()
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug

    assert len(physical._raw_interactions[constants.INTERACTION_LINEAR]) == 1
    assert len(physical._raw_interactions[constants.INTERACTION_QUADRATIC]) == 0

    assert not ising._interactions[ising._interactions["name"] == "x[0]"]["removed"].values[0]
    assert not ising._interactions[ising._interactions["name"] == "x[0]"]["dirty"].values[0]
    assert len(ising._interactions[ising._interactions["name"] == "x[1]"]) == 0


@pytest.mark.parametrize("n,s", [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (10, 100)])
def test_logical_model_to_physical_with_n_hot_constraint_qubo(n, s):
    # n out of s variables should be 1
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(s,))
    model.n_hot_constraint(x, n=n)
    physical = model.to_physical()

    for i in range(s):
        assert physical._raw_interactions[constants.INTERACTION_LINEAR][f"x[{i}]"] == 2 * n - 1.0
    for i in range(s):
        for j in range(s):
            l1 = f"x[{i}]"
            l2 = f"x[{j}]"
            if l1 < l2:
                assert physical._raw_interactions[constants.INTERACTION_QUADRATIC][(l1, l2)] == -2.0


@pytest.mark.parametrize("n,s", [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (10, 100)])
def test_logical_model_to_physical_with_n_hot_constraint_ising(n, s):
    # n out of s spins should be +1
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(s,))
    model.n_hot_constraint(x, n=n)
    physical = model.to_physical()

    for i in range(s):
        if s != 2 * n:
            assert physical._raw_interactions[constants.INTERACTION_LINEAR][f"x[{i}]"] == -1.0 * (s - 2 * n)
        else:
            assert f"x[{i}]" not in physical._raw_interactions[constants.INTERACTION_LINEAR]
    for i in range(s):
        for j in range(s):
            l1 = f"x[{i}]"
            l2 = f"x[{j}]"
            if l1 < l2:
                assert physical._raw_interactions[constants.INTERACTION_QUADRATIC][(l1, l2)] == -1.0


def test_logical_model_to_physical_with_n_hot_constraint_randomly_qubo(qubo):
    x = qubo.variables("x", shape=(4,))
    qubo.n_hot_constraint(x[(slice(0, 2),)], n=1, strength=10)
    qubo.n_hot_constraint(x[1], n=1, strength=10)
    qubo.n_hot_constraint(x[2], n=1, strength=10)
    qubo.n_hot_constraint(x, n=1, strength=10)
    physical = qubo.to_physical()

    for i in range(3):
        assert physical._raw_interactions[constants.INTERACTION_LINEAR][f"x[{i}]"] == 10.0
    for i in range(2):
        for j in range(i + 1, 3):
            assert physical._raw_interactions[constants.INTERACTION_QUADRATIC][(f"x[{i}]", f"x[{j}]")] == -20.0


def test_logical_model_to_physical_with_n_hot_constraint_randomly_ising(ising):
    x = ising.variables("x", shape=(4,))
    ising.n_hot_constraint(x[(slice(0, 2),)], n=1)
    physical = ising.to_physical()
    ising.n_hot_constraint(x[1], n=1)
    physical = ising.to_physical()
    ising.n_hot_constraint(x[2], n=1)
    physical = ising.to_physical()
    ising.n_hot_constraint(x, n=1)
    physical = ising.to_physical()

    for i in range(3):
        assert physical._raw_interactions[constants.INTERACTION_LINEAR][f"x[{i}]"] == -2.0
    for i in range(2):
        for j in range(i + 1, 3):
            assert physical._raw_interactions[constants.INTERACTION_QUADRATIC][(f"x[{i}]", f"x[{j}]")] == -1.0


def test_logical_model_to_physical_with_placeholder(ising):
    placeholder = {"a": 10.0}
    ising.to_physical(placeholder=placeholder)
    # TODO


def test_logical_model_to_physical_with_deleted_variables(ising):
    x = ising.variables("x", shape=(3,))
    y = ising.variables("y", shape=(2, 2))  # noqa: F841
    ising.delete_variable(x[0])
    physical = ising.to_physical()

    assert physical._label_to_index["x[1]"] == 0
    assert physical._label_to_index["x[2]"] == 1
    assert physical._label_to_index["y[0][0]"] == 2
    assert physical._label_to_index["y[0][1]"] == 3
    assert physical._label_to_index["y[1][0]"] == 4
    assert physical._label_to_index["y[1][1]"] == 5
    assert len(physical._label_to_index) == 6

    assert physical._index_to_label[0] == "x[1]"
    assert physical._index_to_label[1] == "x[2]"
    assert physical._index_to_label[2] == "y[0][0]"
    assert physical._index_to_label[3] == "y[0][1]"
    assert physical._index_to_label[4] == "y[1][0]"
    assert physical._index_to_label[5] == "y[1][1]"
    assert len(physical._index_to_label) == 6


def test_logical_model_to_physical_with_fixed_variables(ising):
    x = ising.variables("x", shape=(2,))
    ising.add_interaction(x[0], coefficient=100.0)
    ising.fix_variable(x[0], 1)
    physical = ising.to_physical()

    assert physical._label_to_index["x[0]"] == 0
    assert physical._label_to_index["x[1]"] == 1
    assert len(physical._label_to_index) == 2

    assert physical._index_to_label[0] == "x[0]"
    assert physical._index_to_label[1] == "x[1]"
    assert len(physical._index_to_label) == 2

    assert physical._offset == -100.0


def test_logical_model_convert_model_type(qubo):
    x = qubo.variables("x", shape=(3,))
    y = qubo.variables("y", shape=(2, 2))
    qubo.add_interaction(x[0], coefficient=10.0)
    qubo.add_interaction(x[1], coefficient=11.0)
    qubo.add_interaction((x[1], x[2]), coefficient=12.0)
    qubo.add_interaction(y[0, 0], coefficient=-20.0)
    qubo.add_interaction(y[1, 1], coefficient=-22.0)
    qubo.remove_interaction(y[0, 0])

    # To Ising
    qubo.to_ising()

    # - Check variables
    assert qubo._mtype == constants.MODEL_ISING
    assert len(qubo.get_variables()) == 2
    assert isinstance(qubo.get_variables()["x"], pyqubo.Array)
    assert isinstance(qubo.get_variables()["y"], pyqubo.Array)
    assert isinstance(qubo.get_variables()["x"][0], pyqubo.Spin)
    assert isinstance(qubo.get_variables()["y"][0, 0], pyqubo.Spin)

    # - Check interactions
    assert len(qubo._interactions_array["name"]) == 7
    assert qubo._interactions_length == 7

    assert qubo._interactions_array["name"][0] == "x[0]"
    assert qubo._interactions_array["coefficient"][0] == 5.0
    assert qubo._interactions_array["name"][1] == "x[1]"
    assert qubo._interactions_array["coefficient"][1] == 5.5
    assert qubo._interactions_array["name"][2] == "x[1]*x[2]"
    assert qubo._interactions_array["coefficient"][2] == 3.0
    assert qubo._interactions_array["name"][3] == "y[0][0]"
    assert qubo._interactions_array["removed"][3]
    assert qubo._interactions_array["name"][4] == "y[1][1]"
    assert qubo._interactions_array["coefficient"][4] == -11.0
    assert "x[1] from x[1]*x[2] (mtype additional" in qubo._interactions_array["name"][5]
    assert qubo._interactions_array["coefficient"][5] == 3.0
    assert "x[2] from x[1]*x[2] (mtype additional" in qubo._interactions_array["name"][6]
    assert qubo._interactions_array["coefficient"][6] == 3.0

    # - Check offset
    assert qubo._offset == 2.5

    # Convert to Physical
    physical = qubo.to_physical()

    assert physical._mtype == constants.MODEL_ISING
    assert len(physical._raw_interactions[constants.INTERACTION_LINEAR]) == 4
    assert len(physical._raw_interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert physical._raw_interactions[constants.INTERACTION_LINEAR]["x[0]"] == 5.0
    assert physical._raw_interactions[constants.INTERACTION_LINEAR]["x[1]"] == 8.5
    assert physical._raw_interactions[constants.INTERACTION_LINEAR]["x[2]"] == 3.0
    assert physical._raw_interactions[constants.INTERACTION_LINEAR]["y[1][1]"] == -11.0
    assert physical._raw_interactions[constants.INTERACTION_QUADRATIC][("x[1]", "x[2]")] == 3.0

    with pytest.warns(UserWarning):
        qubo.to_ising()

    # Re-convert to QUBO
    qubo._convert_mtype()

    # - Check variables
    assert qubo._mtype == constants.MODEL_QUBO
    assert len(qubo.get_variables()) == 2
    assert isinstance(qubo.get_variables()["x"], pyqubo.Array)
    assert isinstance(qubo.get_variables()["y"], pyqubo.Array)
    assert isinstance(qubo.get_variables()["x"][0], pyqubo.Binary)
    assert isinstance(qubo.get_variables()["y"][0, 0], pyqubo.Binary)

    # - Check interactions
    assert len(qubo._interactions_array["name"]) == 8
    assert qubo._interactions_length == 8

    assert qubo._interactions_array["name"][0] == "x[0]"
    assert qubo._interactions_array["coefficient"][0] == 10.0
    assert qubo._interactions_array["name"][1] == "x[1]"
    assert qubo._interactions_array["coefficient"][1] == 11.0
    assert qubo._interactions_array["name"][2] == "x[1]*x[2]"
    assert qubo._interactions_array["coefficient"][2] == 12.0
    assert qubo._interactions_array["name"][3] == "y[1][1]"
    assert qubo._interactions_array["coefficient"][3] == -22.0
    assert "x[1] from x[1]*x[2] (mtype additional" in qubo._interactions_array["name"][4]
    assert qubo._interactions_array["coefficient"][4] == 6.0
    assert "x[2] from x[1]*x[2] (mtype additional" in qubo._interactions_array["name"][5]
    assert qubo._interactions_array["coefficient"][5] == 6.0
    assert "x[1] from x[1]*x[2] (mtype additional" in qubo._interactions_array["name"][6]
    assert qubo._interactions_array["coefficient"][6] == -6.0
    assert "x[2] from x[1]*x[2] (mtype additional" in qubo._interactions_array["name"][7]
    assert qubo._interactions_array["coefficient"][7] == -6.0

    # - Check offset
    assert qubo._offset == 0.0

    # Convert to Physical
    physical = qubo.to_physical()

    assert physical._mtype == constants.MODEL_QUBO
    assert len(physical._raw_interactions[constants.INTERACTION_LINEAR]) == 3
    assert len(physical._raw_interactions[constants.INTERACTION_QUADRATIC]) == 1
    assert physical._raw_interactions[constants.INTERACTION_LINEAR]["x[0]"] == 10.0
    assert physical._raw_interactions[constants.INTERACTION_LINEAR]["x[1]"] == 11.0
    assert physical._raw_interactions[constants.INTERACTION_LINEAR]["y[1][1]"] == -22.0
    assert physical._raw_interactions[constants.INTERACTION_QUADRATIC][("x[1]", "x[2]")] == 12.0

    with pytest.warns(UserWarning):
        qubo.to_qubo()


@pytest.fixture
def ising_x22():
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(2, 2))
    model.add_interaction(x[0, 0], coefficient=10.0)
    model.add_interaction((x[0, 0], x[1, 1]), coefficient=11.0, attributes={"foo1": "bar1", "myattr": "mymy"})
    return model


@pytest.fixture
def qubo_a22():
    model = LogicalModel(mtype="qubo")
    a = model.variables("a", shape=(2, 2))
    model.add_interaction(a[0, 0], coefficient=10.0)
    model.add_interaction((a[0, 0], a[1, 1]), coefficient=11.0)
    return model


@pytest.fixture
def ising_y22():
    model = LogicalModel(mtype="ising")
    y = model.variables("y", shape=(2, 2))
    model.add_interaction(y[0, 0], coefficient=10.0)
    model.add_interaction((y[0, 0], y[1, 1]), coefficient=11.0)
    return model


@pytest.fixture
def ising_z3():
    model = LogicalModel(mtype="ising")
    z = model.variables("z", shape=(3,))
    model.n_hot_constraint(target=z, n=1)
    return model


@pytest.fixture
def ising_x44():
    model = LogicalModel(mtype="ising")
    x = model.variables("x", shape=(4, 4))
    model.add_interaction(x[0, 0], coefficient=20.0)
    model.add_interaction((x[1, 1], x[2, 2]), coefficient=21.0, attributes={"foo2": "bar2"})
    model.add_interaction(x[3, 3], coefficient=22.0, scale=2.0, timestamp=12345, attributes={"myattr": "mymymymy"})
    return model


@pytest.fixture
def ising_x999():
    model = LogicalModel(mtype="ising")
    model.variables("x", shape=(9, 9, 9))
    return model


def test_logical_model_merge_x22_y22(ising_x22, ising_y22):
    ising_x22.merge(ising_y22)
    _check_merge_of_x22_y22(ising_x22)


def test_logical_model_merge_y22_x22(ising_y22, ising_x22):
    ising_y22.merge(ising_x22)
    _check_merge_of_x22_y22(ising_y22)


def _check_merge_of_x22_y22(model):
    # Check variables
    assert "x" in model._variables
    assert "y" in model._variables
    assert model._variables["x"].shape == (2, 2)
    assert model._variables["y"].shape == (2, 2)

    # Check interactions
    assert model._interactions_length == 4
    for key, interaction in model._interactions_array.items():
        assert len(interaction) == 4
    assert len(model.select_interaction("key == 'x[0][0]'")) == 1
    assert model.select_interaction("key == 'x[0][0]'")["coefficient"].values[0] == 10.0
    assert model.select_interaction("name == 'x[0][0]*x[1][1]'")["attributes.foo1"].values[0] == "bar1"
    assert model.select_interaction("name == 'x[0][0]*x[1][1]'")["attributes.myattr"].values[0] == "mymy"

    assert model.get_offset() == 0.0
    assert model.get_deleted_size() == 0
    assert model.get_fixed_size() == 0


def test_logical_model_merge_x22_x44(ising_x22, ising_x44):
    ising_x22.merge(ising_x44)
    _check_merge_of_x22_x44(ising_x22)


def test_logical_model_merge_x44_x22(ising_x44, ising_x22):
    ising_x44.merge(ising_x22)
    _check_merge_of_x22_x44(ising_x44)


def _check_merge_of_x22_x44(model):
    # Check variables
    assert "x" in model._variables
    assert "y" not in model._variables
    assert model._variables["x"].shape == (4, 4)

    # Check interactions
    assert model._interactions_length == 5
    for key, interaction in model._interactions_array.items():
        assert len(interaction) == 5
    assert len(model.select_interaction("key == 'x[0][0]'")) == 2
    assert (model.select_interaction("key == 'x[0][0]'")["coefficient"].values[0]) + (
        model.select_interaction("key == 'x[0][0]'")["coefficient"].values[1]
    ) == 10.0 + 20.0
    assert model.select_interaction("name == 'x[0][0]*x[1][1]'")["attributes.foo1"].values[0] == "bar1"
    assert np.isnan(model.select_interaction("name == 'x[0][0]*x[1][1]'")["attributes.foo2"].values[0])
    assert np.isnan(model.select_interaction("name == 'x[1][1]*x[2][2]'")["attributes.foo1"].values[0])
    assert model.select_interaction("name == 'x[1][1]*x[2][2]'")["attributes.foo2"].values[0] == "bar2"
    assert model.select_interaction("name == 'x[3][3]'")["scale"].values[0] == 2.0
    assert model.select_interaction("name == 'x[3][3]'")["timestamp"].values[0] == 12345
    assert model.select_interaction("name == 'x[0][0]*x[1][1]'")["attributes.myattr"].values[0] == "mymy"
    assert model.select_interaction("name == 'x[3][3]'")["attributes.myattr"].values[0] == "mymymymy"

    assert model.get_offset() == 0.0
    assert model.get_deleted_size() == 0
    assert model.get_fixed_size() == 0


def test_logical_model_merge_with_constraints(ising_x22, ising_z3):
    ising_x22.merge(ising_z3)

    assert len(ising_x22.get_constraints()) == 1
    assert "Default N-hot Constraint" in ising_x22.get_constraints()
    assert ising_x22.get_constraints_by_label("Default N-hot Constraint")._n == 1
    assert len(ising_x22.get_constraints_by_label("Default N-hot Constraint")._variables) == 3
    assert len(ising_x22._interactions_array["name"]) == 2 + 6


def test_logical_model_merge_with_constraints_both(ising_x22, ising_z3):
    z = ising_x22.variables("z", shape=(2,))
    ising_x22.n_hot_constraint(z, n=2, label="my label")
    ising_x22.merge(ising_z3)

    assert len(ising_x22._constraints) == 2
    assert "Default N-hot Constraint" in ising_x22.get_constraints()
    assert "my label" in ising_x22.get_constraints()
    assert ising_x22.get_constraints_by_label("Default N-hot Constraint")._n == 1
    assert len(ising_x22.get_constraints_by_label("Default N-hot Constraint")._variables) == 3
    assert ising_x22.get_constraints_by_label("my label")._n == 2
    assert len(ising_x22.get_constraints_by_label("my label")._variables) == 2
    assert len(ising_x22._interactions_array["name"]) == 2 + 3 + 6


def test_logical_model_merge_with_constraints_both_invalid(ising_x22, ising_z3):
    # Both models have constraints with the same label name, cannot merge.
    z = ising_x22.variables("z", shape=(2,))
    ising_x22.n_hot_constraint(z, n=1)
    with pytest.raises(ValueError):
        ising_x22.merge(ising_z3)


def test_logical_model_merge_with_delete(ising_x22, ising_x44):
    x = ising_x22.get_variables_by_name(name="x")
    ising_x22.delete_variable(x[1, 1])
    x = ising_x44.get_variables_by_name(name="x")
    ising_x44.delete_variable(target=x[3, 3])
    ising_x22.merge(ising_x44)

    assert ising_x22.get_deleted_size() == 2
    assert "x[1][1]" in ising_x22.get_deleted_array()
    assert "x[3][3]" in ising_x22.get_deleted_array()

    assert ising_x22.select_interaction("name == 'x[0][0]*x[1][1]'")["removed"].values[0]
    assert not ising_x22.select_interaction("name == 'x[1][1]*x[2][2]'")["removed"].values[0]
    assert ising_x22.select_interaction("name == 'x[3][3]'")["removed"].values[0]


def test_logical_model_merge_with_fix(ising_x22, ising_x44):
    x = ising_x22.get_variables_by_name(name="x")
    ising_x22.fix_variable(x[1, 1], 1)
    x = ising_x44.get_variables_by_name(name="x")
    ising_x44.fix_variable(target=x[3, 3], value=-1)
    ising_x22.merge(ising_x44)

    assert ising_x22.get_fixed_size() == 2
    assert "x[1][1]" in ising_x22.get_fixed_array()
    assert "x[3][3]" in ising_x22.get_fixed_array()

    assert ising_x22.select_interaction("name == 'x[0][0]*x[1][1]'")["removed"].values[0]
    assert not ising_x22.select_interaction("name == 'x[1][1]*x[2][2]'")["removed"].values[0]
    assert ising_x22.select_interaction("name == 'x[3][3]'")["removed"].values[0]
    assert ising_x22.select_interaction("name == 'x[0][0] (before fixed: x[0][0]*x[1][1])'")["coefficient"].values[0]
    assert ising_x22.get_offset() == 22.0 * 2.0


def test_logical_model_merge_x22_a22_invalid(ising_x22, qubo_a22):
    with pytest.raises(ValueError):
        ising_x22.merge(qubo_a22)

    with pytest.raises(ValueError):
        qubo_a22.merge(ising_x22)


def test_logical_model_merge_x22_x999_invalid(ising_x22, ising_x999):
    with pytest.raises(ValueError):
        ising_x22.merge(ising_x999)

    with pytest.raises(ValueError):
        ising_x999.merge(other=ising_x22)

    with pytest.raises(TypeError):
        ising_x999.merge("other type")


def test_logical_model_merge_physical_invalid(ising_x22, ising_x44):
    physical = ising_x44.to_physical()
    with pytest.raises(TypeError):
        ising_x22.merge(physical)
