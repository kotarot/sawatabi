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
import pandas as pd
import pytest

from sawatabi.model import LogicalModel


@pytest.fixture
def ising():
    return LogicalModel(mtype="ising")


@pytest.fixture
def qubo():
    return LogicalModel(mtype="qubo")


################################
# Select
################################


def test_logical_model_select(ising):
    x = ising.variables("x", shape=(10, 10))
    ising.add_interaction(x[0][0], coefficient=10.0)
    ising.add_interaction(x[0, 1], name="my name", coefficient=20.0)
    ising.add_interaction((x[0, 0], x[0, 1]), coefficient=30.0, timestamp=1234567890.123, attributes={"foo": "bar", "my attr": "my value"})

    # single result
    selected = ising.select_interaction("name == 'x[0][0]'")
    assert type(selected) == pd.core.frame.DataFrame
    assert len(selected) == 1
    assert selected["name"].values[0] == "x[0][0]"
    assert selected["key"].values[0] == "x[0][0]"
    assert id(selected["interacts"].values[0]) == id(x[0][0])
    assert selected["coefficient"].values[0] == 10.0

    # dict format
    selected = ising.select_interaction("name == 'my name'", fmt="dict")
    assert type(selected) == dict
    assert len(selected) == 1
    key = list(selected.keys())[0]
    assert selected[key]["name"] == "my name"
    assert selected[key]["key"] == "x[0][1]"
    assert id(selected[key]["interacts"]) == id(x[0][1])
    assert selected[key]["coefficient"] == 20.0

    # multiple results
    selected = ising.select_interaction("timestamp > 1234567890.000")
    assert len(selected) == 3
    assert selected["name"].values[0] == "x[0][0]"
    assert selected["name"].values[1] == "my name"
    assert selected["name"].values[2] == "x[0][0]*x[0][1]"
    assert selected["coefficient"].values[0] == 10.0
    assert selected["coefficient"].values[1] == 20.0
    assert selected["coefficient"].values[2] == 30.0
    assert selected["attributes.foo"].values[2] == "bar"
    assert selected["attributes.my attr"].values[2] == "my value"

    # empty
    selected = ising.select_interaction("timestamp < 1234567890.000")
    assert len(selected) == 0

    # attributes
    selected = ising.select_interaction("`attributes.foo` == 'bar'")
    assert len(selected) == 1
    assert selected["name"].values[0] == "x[0][0]*x[0][1]"

    selected = ising.select_interaction("`attributes.my attr` == 'my value'")
    assert len(selected) == 1
    assert selected["name"].values[0] == "x[0][0]*x[0][1]"

    # invalid query
    with pytest.raises(pd.core.computation.ops.UndefinedVariableError):
        ising.select_interaction("invalid == 'invalid'")

    # invalid format
    with pytest.raises(ValueError):
        ising.select_interaction("name == 'x[0][0]'", fmt="invalid")


def test_logical_model_select_interactions_by_variable(ising):
    x = ising.variables("x", shape=(10, 10))
    ising.add_interaction(x[0, 0], coefficient=10.0)
    ising.add_interaction(x[0, 1], coefficient=20.0)
    ising.add_interaction((x[0, 0], x[0, 1]), coefficient=30.0)

    selected = ising.select_interactions_by_variable(x[0, 0])
    assert type(selected) == np.ndarray
    assert len(selected) == 2
    assert selected[0] == "x[0][0]"
    assert selected[1] == "x[0][0]*x[0][1]"


################################
# Add
################################


def test_logical_model_add(ising):
    x = ising.variables("x", shape=(2, 2))

    ising.add_interaction(x[0, 0], coefficient=1.0)
    assert len(ising._interactions_array["name"]) == 1
    assert ising._interactions_array["name"][0] == "x[0][0]"
    assert len(ising._interactions_array["key"]) == 1
    assert ising._interactions_array["key"][0] == "x[0][0]"
    assert len(ising._interactions_array["interacts"]) == 1
    assert ising._interactions_array["interacts"][0] == x[0, 0]
    assert len(ising._interactions_array["coefficient"]) == 1
    assert ising._interactions_array["coefficient"][0] == 1.0
    assert len(ising._interactions_array["scale"]) == 1
    assert ising._interactions_array["scale"][0] == 1.0
    assert ising._interactions_length == 1

    selected = ising.select_interaction("name == 'x[0][0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 1
    assert selected["name"].values[0] == "x[0][0]"
    assert selected["key"].values[0] == "x[0][0]"
    assert selected["interacts"].values[0] == x[0, 0]
    assert id(selected["interacts"].values[0]) == id(x[0, 0])
    assert ising._interactions[ising._interactions["name"] == "x[0][0]"]["coefficient"].values[0] == 1.0
    assert ising._interactions[ising._interactions["name"] == "x[0][0]"]["scale"].values[0] == 1.0

    ising.add_interaction(x[0, 1], coefficient=2.0, scale=0.1)
    selected = ising.select_interaction("name == 'x[0][1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 2
    assert selected["name"].values[0] == "x[0][1]"
    assert ising._interactions[ising._interactions["name"] == "x[0][1]"]["coefficient"].values[0] == 2.0
    assert ising._interactions[ising._interactions["name"] == "x[0][1]"]["scale"].values[0] == 0.1

    # attributes
    ising.add_interaction(x[1, 0], coefficient=3.0, attributes={"foo": "bar"})
    selected = ising.select_interaction("name == 'x[1][0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 3
    assert selected["name"].values[0] == "x[1][0]"
    assert ising._interactions[ising._interactions["name"] == "x[1][0]"]["coefficient"].values[0] == 3.0
    assert ising._interactions[ising._interactions["name"] == "x[1][0]"]["attributes.foo"].values[0] == "bar"

    # timestamp
    ising.add_interaction((x[0, 0], x[0, 1]), coefficient=-4.0, timestamp=1234567890.123)
    selected = ising.select_interaction("name == 'x[0][0]*x[0][1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 4
    assert selected["name"].values[0] == "x[0][0]*x[0][1]"
    assert selected["key"].values[0] == ("x[0][0]", "x[0][1]")
    assert selected["interacts"].values[0] == (x[0, 0], x[0, 1])
    assert id(selected["interacts"].values[0][0]) == id(x[0, 0])
    assert id(selected["interacts"].values[0][1]) == id(x[0, 1])
    assert ising._interactions[ising._interactions["name"] == "x[0][0]*x[0][1]"]["coefficient"].values[0] == -4.0
    assert ising._interactions[ising._interactions["name"] == "x[0][0]*x[0][1]"]["timestamp"].values[0] == 1234567890.123

    # Test key order
    ising.add_interaction((x[1, 1], x[1, 0]), coefficient=-4.0, timestamp=1234567890.123)
    selected = ising.select_interaction("name == 'x[1][0]*x[1][1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 5
    assert selected["name"].values[0] == "x[1][0]*x[1][1]"
    assert selected["key"].values[0] == ("x[1][0]", "x[1][1]")
    assert selected["interacts"].values[0] == (x[1, 0], x[1, 1])
    assert id(selected["interacts"].values[0][0]) == id(x[1, 0])
    assert id(selected["interacts"].values[0][1]) == id(x[1, 1])
    assert ising._interactions[ising._interactions["name"] == "x[1][0]*x[1][1]"]["coefficient"].values[0] == -4.0
    assert ising._interactions[ising._interactions["name"] == "x[1][0]*x[1][1]"]["timestamp"].values[0] == 1234567890.123


def test_logical_model_add_invalid_arguments(ising):
    x = ising.variables("x", shape=(3,))
    y = ising.variables("y", shape=(2,))

    with pytest.raises(ValueError):
        ising.add_interaction(target=None)

    with pytest.raises(TypeError):
        ising.add_interaction()

    with pytest.raises(TypeError):
        ising.add_interaction("invalid type", coefficient=1.0)

    with pytest.raises(TypeError):
        ising.add_interaction((x[0], x[1], x[2]), coefficient=1.0)

    with pytest.raises(TypeError):
        ising.add_interaction(("a", "b"), coefficient=1.0)

    with pytest.raises(ValueError):
        ising.add_interaction((x[0], x[0]), coefficient=1.0)

    # Invalid types
    with pytest.raises(TypeError):
        ising.add_interaction(x[0], coefficient="invalid type")

    with pytest.raises(TypeError):
        ising.add_interaction(x[0], scale="invalid type")

    with pytest.raises(TypeError):
        ising.add_interaction(x[0], attributes="invalid type")

    with pytest.raises(TypeError):
        ising.add_interaction(x[0], timestamp="invalid type")

    # Already added
    with pytest.raises(ValueError):
        ising.add_interaction(y[0], coefficient=2.0)
        ising.add_interaction(y[0], coefficient=2.0)

    # Already removed
    with pytest.raises(ValueError):
        ising.add_interaction(y[1], coefficient=2.0)
        ising.remove_interaction(y[1])
        ising.add_interaction(y[1], coefficient=2.0)


def test_logical_model_add_duplicate(ising):
    x = ising.variables("x", shape=(2,))

    ising.add_interaction(x[0], coefficient=1.0)
    selected = ising.select_interaction("name == 'x[0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 1
    assert selected["name"].values[0] == "x[0]"

    ising.add_interaction(x[0], name="my name", coefficient=1.0)
    selected = ising.select_interaction("name == 'my name'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 2
    assert selected["name"].values[0] == "my name"


def test_logical_model_add_duplicate_invalid(ising):
    x = ising.variables("x", shape=(2,))
    ising.add_interaction(x[0], coefficient=1.0)
    ising.add_interaction(x[1], name="my name", coefficient=1.0)

    with pytest.raises(ValueError):
        ising.add_interaction(x[0], coefficient=2.0)

    with pytest.raises(ValueError):
        ising.add_interaction(x[1], name="my name", coefficient=2.0)


################################
# Update
################################


def test_logical_model_update(ising):
    x = ising.variables("x", shape=(2,))

    # initialize
    ising.add_interaction(x[0], coefficient=1.0)
    ising.add_interaction((x[0], x[1]), coefficient=2.0)
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == "x[0]"]["coefficient"].values[0] == 1.0
    assert ising._interactions[ising._interactions["name"] == "x[0]*x[1]"]["coefficient"].values[0] == 2.0

    # update by a variable
    ising.update_interaction(x[0], coefficient=10.0)
    selected = ising.select_interaction("name == 'x[0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == "x[0]"]["coefficient"].values[0] == 10.0
    assert selected["name"].values[0] == "x[0]"
    assert selected["key"].values[0] == "x[0]"
    assert selected["interacts"].values[0] == x[0]
    assert id(selected["interacts"].values[0]) == id(x[0])

    # update by a target
    ising.update_interaction(target=x[0], coefficient=100.0)
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == "x[0]"]["coefficient"].values[0] == 100.0

    # update by a name
    ising.update_interaction(name="x[0]", coefficient=1000.0, scale=2.0, attributes={"foo": "bar"})
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == "x[0]"]["coefficient"].values[0] == 1000.0
    assert ising._interactions[ising._interactions["name"] == "x[0]"]["scale"].values[0] == 2.0
    assert ising._interactions[ising._interactions["name"] == "x[0]"]["attributes.foo"].values[0] == "bar"

    # update by a pair of variables
    ising.update_interaction(target=(x[0], x[1]), coefficient=20.0)
    selected = ising.select_interaction("name == 'x[0]*x[1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == "x[0]*x[1]"]["coefficient"].values[0] == 20.0
    assert selected["name"].values[0] == "x[0]*x[1]"
    assert selected["key"].values[0] == ("x[0]", "x[1]")
    assert selected["interacts"].values[0] == (x[0], x[1])
    assert id(selected["interacts"].values[0][0]) == id(x[0])
    assert id(selected["interacts"].values[0][1]) == id(x[1])

    # update by a pair of variables (reversed order)
    ising.update_interaction(target=(x[1], x[0]), coefficient=200.0)
    selected = ising.select_interaction("name == 'x[0]*x[1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == "x[0]*x[1]"]["coefficient"].values[0] == 200.0
    assert selected["name"].values[0] == "x[0]*x[1]"
    assert selected["key"].values[0] == ("x[0]", "x[1]")
    assert selected["interacts"].values[0] == (x[0], x[1])
    assert id(selected["interacts"].values[0][0]) == id(x[0])
    assert id(selected["interacts"].values[0][1]) == id(x[1])

    # update by a name
    ising.update_interaction(name="x[0]*x[1]", coefficient=2000.0)
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == "x[0]*x[1]"]["coefficient"].values[0] == 2000.0


def test_logical_model_update_by_custom_names(ising):
    y = ising.variables("y", shape=(2,))
    n1 = "custom interaction 1"
    n2 = "custom interaction 2"
    n3 = "custom interaction 3"

    # initialize
    ising.add_interaction(y[0], coefficient=-1.0, name=n1)
    ising.add_interaction((y[0], y[1]), coefficient=-2.0, name=n2)
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == n1]["coefficient"].values[0] == -1.0
    assert ising._interactions[ising._interactions["name"] == n2]["coefficient"].values[0] == -2.0

    ising.update_interaction(name=n1, coefficient=-10.0)
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == n1]["coefficient"].values[0] == -10.0

    ising.update_interaction(name=n2, coefficient=-20.0)
    ising._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == n2]["coefficient"].values[0] == -20.0

    with pytest.raises(KeyError):
        ising.update_interaction(name=n3, coefficient=-30.0)


def test_logical_model_update_without_initialize(ising):
    # The following operation will be successful with a UserWarning.
    x = ising.variables("x", shape=(3,))

    with pytest.warns(UserWarning):
        ising.update_interaction(x[0], coefficient=11.0)

    selected = ising.select_interaction("name == 'x[0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 1
    assert ising._interactions[ising._interactions["name"] == "x[0]"]["coefficient"].values[0] == 11.0
    assert ising._interactions[ising._interactions["name"] == "x[0]"]["scale"].values[0] == 1.0
    assert selected["name"].values[0] == "x[0]"
    assert selected["key"].values[0] == ("x[0]")
    assert selected["interacts"].values[0] == (x[0])

    with pytest.warns(UserWarning):
        ising.update_interaction((x[0], x[1]), coefficient=22.0, scale=33.0)

    selected = ising.select_interaction("name == 'x[0]*x[1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 2
    assert ising._interactions[ising._interactions["name"] == "x[0]*x[1]"]["coefficient"].values[0] == 22.0
    assert ising._interactions[ising._interactions["name"] == "x[0]*x[1]"]["scale"].values[0] == 33.0
    assert selected["name"].values[0] == "x[0]*x[1]"
    assert selected["key"].values[0] == ("x[0]", "x[1]")
    assert selected["interacts"].values[0] == (x[0], x[1])


def test_logical_model_update_invalid(ising):
    x = ising.variables("x", shape=(3,))
    ising.add_interaction(x[0], coefficient=1.0)

    with pytest.raises(ValueError):
        ising.update_interaction()

    with pytest.raises(ValueError):
        ising.update_interaction(x[0], name="x[0]")

    with pytest.raises(KeyError):
        ising.update_interaction(name="x[1]", coefficient=1.0)

    with pytest.raises(TypeError):
        ising.update_interaction("invalid type", coefficient=1.0)

    with pytest.raises(TypeError):
        ising.update_interaction((x[0], x[1], x[2]), coefficient=1.0)

    with pytest.raises(TypeError):
        ising.update_interaction(("a", "b"), coefficient=1.0)

    with pytest.raises(ValueError):
        ising.update_interaction((x[0], x[0]), coefficient=1.0)

    # Invalid types
    with pytest.raises(TypeError):
        ising.update_interaction(x[0], coefficient="invalid type")

    with pytest.raises(TypeError):
        ising.update_interaction(x[0], scale="invalid type")

    with pytest.raises(TypeError):
        ising.update_interaction(x[0], attributes="invalid type")

    with pytest.raises(TypeError):
        ising.update_interaction(x[0], timestamp="invalid type")

    # Already removed
    with pytest.raises(ValueError):
        ising.add_interaction(x[1], coefficient=2.0)
        ising.remove_interaction(x[1])
        ising.update_interaction(x[1], coefficient=2.0)


################################
# Remove
################################


def test_logical_model_remove_by_target(ising):
    x = ising.variables("x", shape=(2,))

    # initialize
    ising.add_interaction(x[0], coefficient=1.0)
    ising.add_interaction(x[1], coefficient=10.0)
    assert ising._interactions_length == 2

    # remove
    ising.remove_interaction(x[0])
    selected = ising.select_interaction("key == 'x[0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 2
    assert selected["name"].values[0] == "x[0]"
    assert selected["key"].values[0] == "x[0]"
    assert selected["coefficient"].values[0] == 1.0
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]

    ising.remove_interaction(name="x[1]")
    selected = ising.select_interaction("key == 'x[1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(ising._interactions) == 2
    assert selected["name"].values[0] == "x[1]"
    assert selected["key"].values[0] == "x[1]"
    assert selected["coefficient"].values[0] == 10.0
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]


def test_logical_model_remove_by_name(ising):
    x = ising.variables("x", shape=(2,))

    # initialize
    ising.add_interaction(x[0], "my name", coefficient=1.0)
    assert ising._interactions_length == 1

    # remove
    ising.remove_interaction(name="my name")
    assert ising._interactions_length == 1
    assert ising.select_interaction("name == 'my name'")["removed"].values[0]


def test_logical_model_remove_invalid(ising):
    x = ising.variables("x", shape=(3,))
    ising.add_interaction(x[0], coefficient=1.0)

    with pytest.raises(ValueError):
        ising.remove_interaction()

    with pytest.raises(ValueError):
        ising.remove_interaction(x[0], name="x[0]")

    with pytest.raises(KeyError):
        ising.remove_interaction(x[1])

    with pytest.raises(TypeError):
        ising.remove_interaction("invalid type")

    with pytest.raises(TypeError):
        ising.update_interaction((x[0], x[1], x[2]))

    with pytest.raises(TypeError):
        ising.update_interaction(("a", "b"))

    with pytest.raises(ValueError):
        ising.update_interaction((x[0], x[0]))


################################
# Delete
################################


def test_logical_model_delete(ising):
    x = ising.variables("x", shape=(2, 3, 4))
    y = ising.variables("y", shape=(5, 6))  # noqa: F841

    ising.add_interaction(x[0, 0, 0], coefficient=1.0)
    assert ising._interactions_length == 1

    ising.add_interaction((x[0, 0, 0], x[0, 0, 1]), coefficient=2.0)
    assert ising._interactions_length == 2

    assert ising.get_size() == 2 * 3 * 4 + 5 * 6
    assert ising.get_deleted_size() == 0
    assert ising.get_all_size() == 2 * 3 * 4 + 5 * 6

    # Set dirty flags for interactions releted to the deleting variable
    ising.delete_variable(x[0, 0, 0])
    assert ising._interactions_length == 2

    assert ising.get_size() == 2 * 3 * 4 + 5 * 6 - 1
    assert ising.get_deleted_size() == 1
    assert ising.get_all_size() == 2 * 3 * 4 + 5 * 6

    # Convert physical to resolve dirty
    ising.to_physical()
    assert len(ising._interactions_array["name"]) == 0
    assert ising._interactions_length == 0

    assert ising.get_size() == 2 * 3 * 4 + 5 * 6 - 1
    assert ising.get_deleted_size() == 1
    assert ising.get_all_size() == 2 * 3 * 4 + 5 * 6


def test_logical_model_delete_dealing_with_nhot_constraints_qubo():
    model = LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(4,))
    default_label = "Default N-hot Constraint"

    model.n_hot_constraint(x, n=1, strength=1.0)
    assert len(model.get_constraints()) == 1
    assert default_label in model.get_constraints()
    assert model.get_constraints_by_label(default_label)._n == 1
    assert len(model.get_constraints_by_label(default_label)._variables) == 4

    model.delete_variable(x[0])

    assert len(model.get_constraints()) == 1
    assert default_label in model.get_constraints()
    assert model.get_constraints_by_label(default_label)._n == 1
    assert len(model.get_constraints_by_label(default_label)._variables) == 3
    assert model._interactions[model._interactions["name"] == f"x[1] ({default_label})"]["coefficient"].values[0] == 1.0
    assert model._interactions[model._interactions["name"] == f"x[1]*x[2] ({default_label})"]["coefficient"].values[0] == -2.0


def test_logical_model_delete_invalid_argument(ising):
    with pytest.raises(TypeError):
        ising.delete_variable()

    with pytest.raises(TypeError):
        ising.delete_variable("invalid type")

    with pytest.raises(ValueError):
        ising.delete_variable(target=None)


################################
# Fix
################################


def test_logical_model_fix_ising(ising):
    x = ising.variables("x", shape=(6,))
    ising.add_interaction(x[0], coefficient=10.0, scale=1.1)
    ising.add_interaction(x[1], coefficient=20.0)
    ising.add_interaction(x[2], coefficient=30.0)
    ising.add_interaction((x[1], x[2]), coefficient=40.0)
    ising.add_interaction((x[1], x[2]), name="my interaction", coefficient=45.0)
    ising.add_interaction(x[3], coefficient=50.0)
    ising.add_interaction(x[4], coefficient=60.0)
    ising.add_interaction((x[3], x[4]), coefficient=70.0)

    with pytest.raises(ValueError):
        ising.fix_variable(target=x[0], value=0)

    # Remove a variable that has no 2-body interactions.
    ising.fix_variable(x[0], 1)
    assert ising.get_fixed_size() == 1
    assert "x[0]" in ising.get_fixed_array()
    assert ising.get_offset() == -11.0
    selected = ising.select_interaction("name == 'x[0]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]

    # Remove variables that has 2-body interactions.
    ising.fix_variable(x[1], -1)
    assert ising.get_fixed_size() == 2
    assert "x[1]" in ising.get_fixed_array()
    assert ising.get_offset() == -11.0 + 20.0
    selected = ising.select_interaction("name == 'x[1]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = ising.select_interaction("name == 'x[1]*x[2]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = ising.select_interaction("name == 'my interaction'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = ising.select_interaction("name == 'x[2] (before fixed: x[1]*x[2])'")
    assert selected["body"].values[0] == 1
    assert selected["key"].values[0] == "x[2]"
    assert selected["coefficient"].values[0] == -40.0
    assert selected["scale"].values[0] == 1.0
    assert selected["dirty"].values[0]
    assert not selected["removed"].values[0]
    selected = ising.select_interaction("name == 'x[2] (before fixed: my interaction)'")
    assert selected["body"].values[0] == 1
    assert selected["key"].values[0] == "x[2]"
    assert selected["coefficient"].values[0] == -45.0
    assert selected["scale"].values[0] == 1.0
    assert selected["dirty"].values[0]
    assert not selected["removed"].values[0]

    ising.fix_variable(x[3], 1)
    assert ising.get_fixed_size() == 3
    assert "x[3]" in ising.get_fixed_array()
    assert ising.get_offset() == -11.0 + 20.0 - 50.0
    selected = ising.select_interaction("name == 'x[3]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = ising.select_interaction("name == 'x[3]*x[4]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = ising.select_interaction("name == 'x[4] (before fixed: x[3]*x[4])'")
    assert selected["body"].values[0] == 1
    assert selected["key"].values[0] == "x[4]"
    assert selected["coefficient"].values[0] == 70.0
    assert selected["scale"].values[0] == 1.0
    assert selected["dirty"].values[0]
    assert not selected["removed"].values[0]

    # Nothing happens
    with pytest.warns(UserWarning):
        ising.fix_variable(x[5], 1)


def test_logical_model_fix_qubo(qubo):
    a = qubo.variables("a", shape=(6,))
    qubo.add_interaction(a[0], coefficient=10.0, scale=1.1)
    qubo.add_interaction(a[1], coefficient=20.0)
    qubo.add_interaction(a[2], coefficient=30.0)
    qubo.add_interaction((a[1], a[2]), coefficient=40.0)
    qubo.add_interaction((a[1], a[2]), name="my interaction", coefficient=45.0)
    qubo.add_interaction(a[3], coefficient=50.0)
    qubo.add_interaction(a[4], coefficient=60.0)
    qubo.add_interaction((a[3], a[4]), coefficient=70.0)

    with pytest.raises(ValueError):
        qubo.fix_variable(target=a[0], value=-1)

    # Remove a variable that has no 2-body interactions.
    qubo.fix_variable(a[0], 1)
    assert qubo.get_fixed_size() == 1
    assert "a[0]" in qubo.get_fixed_array()
    assert qubo.get_offset() == -11.0
    selected = qubo.select_interaction("name == 'a[0]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]

    # Remove variables that has 2-body interactions.
    qubo.fix_variable(a[2], 1)
    assert qubo.get_fixed_size() == 2
    assert "a[2]" in qubo.get_fixed_array()
    assert qubo.get_offset() == -11.0 - 30.0
    selected = qubo.select_interaction("name == 'a[2]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = qubo.select_interaction("name == 'a[1]*a[2]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = qubo.select_interaction("name == 'my interaction'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = qubo.select_interaction("name == 'a[1] (before fixed: a[1]*a[2])'")
    assert selected["body"].values[0] == 1
    assert selected["key"].values[0] == "a[1]"
    assert selected["coefficient"].values[0] == 40.0
    assert selected["scale"].values[0] == 1.0
    assert selected["dirty"].values[0]
    assert not selected["removed"].values[0]
    selected = qubo.select_interaction("name == 'a[1] (before fixed: my interaction)'")
    assert selected["body"].values[0] == 1
    assert selected["key"].values[0] == "a[1]"
    assert selected["coefficient"].values[0] == 45.0
    assert selected["scale"].values[0] == 1.0
    assert selected["dirty"].values[0]
    assert not selected["removed"].values[0]

    qubo.fix_variable(a[4], 0)
    selected = qubo.select_interaction("name == 'a[4]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = qubo.select_interaction("name == 'a[3]*a[4]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]

    # Nothing happens
    with pytest.warns(UserWarning):
        qubo.fix_variable(a[5], 1)


def test_logical_model_fix_invalid_argument(ising):
    with pytest.raises(TypeError):
        ising.fix_variable()

    with pytest.raises(TypeError):
        ising.fix_variable("invalid type")

    with pytest.raises(ValueError):
        ising.fix_variable(target=None, value=1)
