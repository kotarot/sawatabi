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
def model():
    return LogicalModel(mtype="ising")


@pytest.fixture
def model_qubo():
    return LogicalModel(mtype="qubo")


################################
# Select
################################


def test_logical_model_select(model):
    x = model.variables("x", shape=(10, 10))
    model.add_interaction(x[0][0], coefficient=10.0)
    model.add_interaction(x[0, 1], name="my name", coefficient=20.0)
    model.add_interaction((x[0, 0], x[0, 1]), coefficient=30.0, timestamp=1234567890123, attributes={"foo": "bar", "my attr": "my value"})

    # single result
    res = model.select_interaction("name == 'x[0][0]'")
    assert type(res) == pd.core.frame.DataFrame
    assert len(res) == 1
    assert res["name"].values[0] == "x[0][0]"
    assert res["key"].values[0] == "x[0][0]"
    assert id(res["interacts"].values[0]) == id(x[0][0])
    assert res["coefficient"].values[0] == 10.0

    # dict format
    res = model.select_interaction("name == 'my name'", fmt="dict")
    assert type(res) == dict
    assert len(res) == 1
    key = list(res.keys())[0]
    assert res[key]["name"] == "my name"
    assert res[key]["key"] == "x[0][1]"
    assert id(res[key]["interacts"]) == id(x[0][1])
    assert res[key]["coefficient"] == 20.0

    # multiple results
    res = model.select_interaction("timestamp > 1234567890000")
    assert len(res) == 3
    assert res["name"].values[0] == "x[0][0]"
    assert res["name"].values[1] == "my name"
    assert res["name"].values[2] == "x[0][0]*x[0][1]"
    assert res["coefficient"].values[0] == 10.0
    assert res["coefficient"].values[1] == 20.0
    assert res["coefficient"].values[2] == 30.0
    assert res["attributes.foo"].values[2] == "bar"
    assert res["attributes.my attr"].values[2] == "my value"

    # empty
    res = model.select_interaction("timestamp < 1234567890000")
    assert len(res) == 0

    # attributes
    res = model.select_interaction("`attributes.foo` == 'bar'")
    assert len(res) == 1
    assert res["name"].values[0] == "x[0][0]*x[0][1]"

    res = model.select_interaction("`attributes.my attr` == 'my value'")
    assert len(res) == 1
    assert res["name"].values[0] == "x[0][0]*x[0][1]"

    # invalid query
    with pytest.raises(pd.core.computation.ops.UndefinedVariableError):
        res = model.select_interaction("invalid == 'invalid'")

    # invalid format
    with pytest.raises(ValueError):
        model.select_interaction("name == 'x[0][0]'", fmt="invalid")


def test_logical_model_select_interactions_by_variable(model):
    x = model.variables("x", shape=(10, 10))
    model.add_interaction(x[0, 0], coefficient=10.0)
    model.add_interaction(x[0, 1], coefficient=20.0)
    model.add_interaction((x[0, 0], x[0, 1]), coefficient=30.0)

    res = model.select_interactions_by_variable(x[0, 0])
    assert type(res) == np.ndarray
    assert len(res) == 2
    assert res[0] == "x[0][0]"
    assert res[1] == "x[0][0]*x[0][1]"


################################
# Add
################################


def test_logical_model_add(model):
    x = model.variables("x", shape=(2, 2))

    model.add_interaction(x[0, 0], coefficient=1.0)
    assert len(model._interactions_array["name"]) == 1
    assert model._interactions_array["name"][0] == "x[0][0]"
    assert len(model._interactions_array["key"]) == 1
    assert model._interactions_array["key"][0] == "x[0][0]"
    assert len(model._interactions_array["interacts"]) == 1
    assert model._interactions_array["interacts"][0] == x[0, 0]
    assert len(model._interactions_array["coefficient"]) == 1
    assert model._interactions_array["coefficient"][0] == 1.0
    assert len(model._interactions_array["scale"]) == 1
    assert model._interactions_array["scale"][0] == 1.0
    assert model._interactions_length == 1

    res = model.select_interaction("name == 'x[0][0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 1
    assert res["name"].values[0] == "x[0][0]"
    assert res["key"].values[0] == "x[0][0]"
    assert res["interacts"].values[0] == x[0, 0]
    assert id(res["interacts"].values[0]) == id(x[0, 0])
    assert model._interactions[model._interactions["name"] == "x[0][0]"]["coefficient"].values[0] == 1.0
    assert model._interactions[model._interactions["name"] == "x[0][0]"]["scale"].values[0] == 1.0

    model.add_interaction(x[0, 1], coefficient=2.0, scale=0.1)
    res = model.select_interaction("name == 'x[0][1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 2
    assert res["name"].values[0] == "x[0][1]"
    assert model._interactions[model._interactions["name"] == "x[0][1]"]["coefficient"].values[0] == 2.0
    assert model._interactions[model._interactions["name"] == "x[0][1]"]["scale"].values[0] == 0.1

    # attributes
    model.add_interaction(x[1, 0], coefficient=3.0, attributes={"foo": "bar"})
    res = model.select_interaction("name == 'x[1][0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 3
    assert res["name"].values[0] == "x[1][0]"
    assert model._interactions[model._interactions["name"] == "x[1][0]"]["coefficient"].values[0] == 3.0
    assert model._interactions[model._interactions["name"] == "x[1][0]"]["attributes.foo"].values[0] == "bar"

    # timestamp
    model.add_interaction((x[0, 0], x[0, 1]), coefficient=-4.0, timestamp=1234567890123)
    res = model.select_interaction("name == 'x[0][0]*x[0][1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 4
    assert res["name"].values[0] == "x[0][0]*x[0][1]"
    assert res["key"].values[0] == ("x[0][0]", "x[0][1]")
    assert res["interacts"].values[0] == (x[0, 0], x[0, 1])
    assert id(res["interacts"].values[0][0]) == id(x[0, 0])
    assert id(res["interacts"].values[0][1]) == id(x[0, 1])
    assert model._interactions[model._interactions["name"] == "x[0][0]*x[0][1]"]["coefficient"].values[0] == -4.0
    assert model._interactions[model._interactions["name"] == "x[0][0]*x[0][1]"]["timestamp"].values[0] == 1234567890123

    # Test key order
    model.add_interaction((x[1, 1], x[1, 0]), coefficient=-4.0, timestamp=1234567890123)
    res = model.select_interaction("name == 'x[1][0]*x[1][1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 5
    assert res["name"].values[0] == "x[1][0]*x[1][1]"
    assert res["key"].values[0] == ("x[1][0]", "x[1][1]")
    assert res["interacts"].values[0] == (x[1, 0], x[1, 1])
    assert id(res["interacts"].values[0][0]) == id(x[1, 0])
    assert id(res["interacts"].values[0][1]) == id(x[1, 1])
    assert model._interactions[model._interactions["name"] == "x[1][0]*x[1][1]"]["coefficient"].values[0] == -4.0
    assert model._interactions[model._interactions["name"] == "x[1][0]*x[1][1]"]["timestamp"].values[0] == 1234567890123


def test_logical_model_add_invalid_arguments(model):
    x = model.variables("x", shape=(3,))
    y = model.variables("y", shape=(2,))

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

    with pytest.raises(ValueError):
        model.add_interaction((x[0], x[0]), coefficient=1.0)

    # Invalid types
    with pytest.raises(TypeError):
        model.add_interaction(x[0], coefficient="invalid type")

    with pytest.raises(TypeError):
        model.add_interaction(x[0], scale="invalid type")

    with pytest.raises(TypeError):
        model.add_interaction(x[0], attributes="invalid type")

    with pytest.raises(TypeError):
        model.add_interaction(x[0], timestamp="invalid type")

    # Already added
    with pytest.raises(ValueError):
        model.add_interaction(y[0], coefficient=2.0)
        model.add_interaction(y[0], coefficient=2.0)

    # Already removed
    with pytest.raises(ValueError):
        model.add_interaction(y[1], coefficient=2.0)
        model.remove_interaction(y[1])
        model.add_interaction(y[1], coefficient=2.0)


def test_logical_model_add_duplicate(model):
    x = model.variables("x", shape=(2,))

    model.add_interaction(x[0], coefficient=1.0)
    res = model.select_interaction("name == 'x[0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 1
    assert res["name"].values[0] == "x[0]"

    model.add_interaction(x[0], name="my name", coefficient=1.0)
    res = model.select_interaction("name == 'my name'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 2
    assert res["name"].values[0] == "my name"


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
    model._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(model._interactions) == 2
    assert model._interactions[model._interactions["name"] == "x[0]"]["coefficient"].values[0] == 1.0
    assert model._interactions[model._interactions["name"] == "x[0]*x[1]"]["coefficient"].values[0] == 2.0

    # update by a variable
    model.update_interaction(x[0], coefficient=10.0)
    res = model.select_interaction("name == 'x[0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 2
    assert model._interactions[model._interactions["name"] == "x[0]"]["coefficient"].values[0] == 10.0
    assert res["name"].values[0] == "x[0]"
    assert res["key"].values[0] == "x[0]"
    assert res["interacts"].values[0] == x[0]
    assert id(res["interacts"].values[0]) == id(x[0])

    # update by a target
    model.update_interaction(target=x[0], coefficient=100.0)
    model._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(model._interactions) == 2
    assert model._interactions[model._interactions["name"] == "x[0]"]["coefficient"].values[0] == 100.0

    # update by a name
    model.update_interaction(name="x[0]", coefficient=1000.0, scale=2.0, attributes={"foo": "bar"})
    model._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(model._interactions) == 2
    assert model._interactions[model._interactions["name"] == "x[0]"]["coefficient"].values[0] == 1000.0
    assert model._interactions[model._interactions["name"] == "x[0]"]["scale"].values[0] == 2.0
    assert model._interactions[model._interactions["name"] == "x[0]"]["attributes.foo"].values[0] == "bar"

    # update by a pair of variables
    model.update_interaction(target=(x[0], x[1]), coefficient=20.0)
    res = model.select_interaction("name == 'x[0]*x[1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 2
    assert model._interactions[model._interactions["name"] == "x[0]*x[1]"]["coefficient"].values[0] == 20.0
    assert res["name"].values[0] == "x[0]*x[1]"
    assert res["key"].values[0] == ("x[0]", "x[1]")
    assert res["interacts"].values[0] == (x[0], x[1])
    assert id(res["interacts"].values[0][0]) == id(x[0])
    assert id(res["interacts"].values[0][1]) == id(x[1])

    # update by a pair of variables (reversed order)
    model.update_interaction(target=(x[1], x[0]), coefficient=200.0)
    res = model.select_interaction("name == 'x[0]*x[1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 2
    assert model._interactions[model._interactions["name"] == "x[0]*x[1]"]["coefficient"].values[0] == 200.0
    assert res["name"].values[0] == "x[0]*x[1]"
    assert res["key"].values[0] == ("x[0]", "x[1]")
    assert res["interacts"].values[0] == (x[0], x[1])
    assert id(res["interacts"].values[0][0]) == id(x[0])
    assert id(res["interacts"].values[0][1]) == id(x[1])

    # update by a name
    model.update_interaction(name="x[0]*x[1]", coefficient=2000.0)
    model._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(model._interactions) == 2
    assert model._interactions[model._interactions["name"] == "x[0]*x[1]"]["coefficient"].values[0] == 2000.0


def test_logical_model_update_by_custom_names(model):
    y = model.variables("y", shape=(2,))
    n1 = "custom interaction 1"
    n2 = "custom interaction 2"
    n3 = "custom interaction 3"

    # initialize
    model.add_interaction(y[0], coefficient=-1.0, name=n1)
    model.add_interaction((y[0], y[1]), coefficient=-2.0, name=n2)
    model._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(model._interactions) == 2
    assert model._interactions[model._interactions["name"] == n1]["coefficient"].values[0] == -1.0
    assert model._interactions[model._interactions["name"] == n2]["coefficient"].values[0] == -2.0

    model.update_interaction(name=n1, coefficient=-10.0)
    model._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(model._interactions) == 2
    assert model._interactions[model._interactions["name"] == n1]["coefficient"].values[0] == -10.0

    model.update_interaction(name=n2, coefficient=-20.0)
    model._update_interactions_dataframe_from_arrays()  # Update the interactions DataFrame for debug
    assert len(model._interactions) == 2
    assert model._interactions[model._interactions["name"] == n2]["coefficient"].values[0] == -20.0

    with pytest.raises(KeyError):
        model.update_interaction(name=n3, coefficient=-30.0)


def test_logical_model_update_without_initialize(model):
    # The following operation will be successful with a UserWarning.
    x = model.variables("x", shape=(3,))

    with pytest.warns(UserWarning):
        model.update_interaction(x[0], coefficient=11.0)

    res = model.select_interaction("name == 'x[0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 1
    assert model._interactions[model._interactions["name"] == "x[0]"]["coefficient"].values[0] == 11.0
    assert model._interactions[model._interactions["name"] == "x[0]"]["scale"].values[0] == 1.0
    assert res["name"].values[0] == "x[0]"
    assert res["key"].values[0] == ("x[0]")
    assert res["interacts"].values[0] == (x[0])

    with pytest.warns(UserWarning):
        model.update_interaction((x[0], x[1]), coefficient=22.0, scale=33.0)

    res = model.select_interaction("name == 'x[0]*x[1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 2
    assert model._interactions[model._interactions["name"] == "x[0]*x[1]"]["coefficient"].values[0] == 22.0
    assert model._interactions[model._interactions["name"] == "x[0]*x[1]"]["scale"].values[0] == 33.0
    assert res["name"].values[0] == "x[0]*x[1]"
    assert res["key"].values[0] == ("x[0]", "x[1]")
    assert res["interacts"].values[0] == (x[0], x[1])


def test_logical_model_update_invalid(model):
    x = model.variables("x", shape=(3,))
    model.add_interaction(x[0], coefficient=1.0)

    with pytest.raises(ValueError):
        model.update_interaction()

    with pytest.raises(ValueError):
        model.update_interaction(x[0], name="x[0]")

    with pytest.raises(KeyError):
        model.update_interaction(name="x[1]", coefficient=1.0)

    with pytest.raises(TypeError):
        model.update_interaction("invalid type", coefficient=1.0)

    with pytest.raises(TypeError):
        model.update_interaction((x[0], x[1], x[2]), coefficient=1.0)

    with pytest.raises(TypeError):
        model.update_interaction(("a", "b"), coefficient=1.0)

    with pytest.raises(ValueError):
        model.update_interaction((x[0], x[0]), coefficient=1.0)

    # Invalid types
    with pytest.raises(TypeError):
        model.update_interaction(x[0], coefficient="invalid type")

    with pytest.raises(TypeError):
        model.update_interaction(x[0], scale="invalid type")

    with pytest.raises(TypeError):
        model.update_interaction(x[0], attributes="invalid type")

    with pytest.raises(TypeError):
        model.update_interaction(x[0], timestamp="invalid type")

    # Already removed
    with pytest.raises(ValueError):
        model.add_interaction(x[1], coefficient=2.0)
        model.remove_interaction(x[1])
        model.update_interaction(x[1], coefficient=2.0)


################################
# Remove
################################


def test_logical_model_remove_by_target(model):
    x = model.variables("x", shape=(2,))

    # initialize
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=10.0)
    assert model._interactions_length == 2

    # remove
    model.remove_interaction(x[0])
    res = model.select_interaction("key == 'x[0]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 2
    assert res["name"].values[0] == "x[0]"
    assert res["key"].values[0] == "x[0]"
    assert res["coefficient"].values[0] == 1.0
    assert res["dirty"].values[0]
    assert res["removed"].values[0]

    model.remove_interaction(name="x[1]")
    res = model.select_interaction("key == 'x[1]'")  # Side effect: Internal interactions DataFrame is updated
    assert len(model._interactions) == 2
    assert res["name"].values[0] == "x[1]"
    assert res["key"].values[0] == "x[1]"
    assert res["coefficient"].values[0] == 10.0
    assert res["dirty"].values[0]
    assert res["removed"].values[0]


def test_logical_model_remove_by_name(model):
    x = model.variables("x", shape=(2,))

    # initialize
    model.add_interaction(x[0], "my name", coefficient=1.0)
    assert model._interactions_length == 1

    # remove
    model.remove_interaction(name="my name")
    assert model._interactions_length == 1
    assert model.select_interaction("name == 'my name'")["removed"].values[0]


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

    with pytest.raises(ValueError):
        model.update_interaction((x[0], x[0]))


################################
# Delete
################################


def test_logical_model_delete(model):
    x = model.variables("x", shape=(2, 3, 4))
    y = model.variables("y", shape=(5, 6))  # noqa: F841

    model.add_interaction(x[0, 0, 0], coefficient=1.0)
    assert model._interactions_length == 1

    model.add_interaction((x[0, 0, 0], x[0, 0, 1]), coefficient=2.0)
    assert model._interactions_length == 2

    assert model.get_size() == 2 * 3 * 4 + 5 * 6
    assert model.get_deleted_size() == 0
    assert model.get_all_size() == 2 * 3 * 4 + 5 * 6

    # Set dirty flags for interactions releted to the deleting variable
    model.delete_variable(x[0, 0, 0])
    assert model._interactions_length == 2

    assert model.get_size() == 2 * 3 * 4 + 5 * 6 - 1
    assert model.get_deleted_size() == 1
    assert model.get_all_size() == 2 * 3 * 4 + 5 * 6

    # Convert physical to resolve dirty
    model.to_physical()
    assert len(model._interactions_array["name"]) == 0
    assert model._interactions_length == 0

    assert model.get_size() == 2 * 3 * 4 + 5 * 6 - 1
    assert model.get_deleted_size() == 1
    assert model.get_all_size() == 2 * 3 * 4 + 5 * 6


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


def test_logical_model_delete_invalid_argument(model):
    with pytest.raises(TypeError):
        model.delete_variable()

    with pytest.raises(TypeError):
        model.delete_variable("invalid type")

    with pytest.raises(ValueError):
        model.delete_variable(target=None)


################################
# Fix
################################


def test_logical_model_fix_ising(model):
    x = model.variables("x", shape=(6,))
    model.add_interaction(x[0], coefficient=10.0, scale=1.1)
    model.add_interaction(x[1], coefficient=20.0)
    model.add_interaction(x[2], coefficient=30.0)
    model.add_interaction((x[1], x[2]), coefficient=40.0)
    model.add_interaction((x[1], x[2]), name="my interaction", coefficient=45.0)
    model.add_interaction(x[3], coefficient=50.0)
    model.add_interaction(x[4], coefficient=60.0)
    model.add_interaction((x[3], x[4]), coefficient=70.0)

    with pytest.raises(ValueError):
        model.fix_variable(target=x[0], value=0)

    # Remove a variable that has no 2-body interactions.
    model.fix_variable(x[0], 1)
    assert model.get_fixed_size() == 1
    assert "x[0]" in model.get_fixed_array()
    assert model.get_offset() == -11.0
    selected = model.select_interaction("name == 'x[0]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]

    # Remove variables that has 2-body interactions.
    model.fix_variable(x[1], -1)
    assert model.get_fixed_size() == 2
    assert "x[1]" in model.get_fixed_array()
    assert model.get_offset() == -11.0 + 20.0
    selected = model.select_interaction("name == 'x[1]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = model.select_interaction("name == 'x[1]*x[2]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = model.select_interaction("name == 'my interaction'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = model.select_interaction("name == 'x[2] (before fixed: x[1]*x[2])'")
    assert selected["body"].values[0] == 1
    assert selected["key"].values[0] == "x[2]"
    assert selected["coefficient"].values[0] == -40.0
    assert selected["scale"].values[0] == 1.0
    assert selected["dirty"].values[0]
    assert not selected["removed"].values[0]
    selected = model.select_interaction("name == 'x[2] (before fixed: my interaction)'")
    assert selected["body"].values[0] == 1
    assert selected["key"].values[0] == "x[2]"
    assert selected["coefficient"].values[0] == -45.0
    assert selected["scale"].values[0] == 1.0
    assert selected["dirty"].values[0]
    assert not selected["removed"].values[0]

    model.fix_variable(x[3], 1)
    assert model.get_fixed_size() == 3
    assert "x[3]" in model.get_fixed_array()
    assert model.get_offset() == -11.0 + 20.0 - 50.0
    selected = model.select_interaction("name == 'x[3]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = model.select_interaction("name == 'x[3]*x[4]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = model.select_interaction("name == 'x[4] (before fixed: x[3]*x[4])'")
    assert selected["body"].values[0] == 1
    assert selected["key"].values[0] == "x[4]"
    assert selected["coefficient"].values[0] == 70.0
    assert selected["scale"].values[0] == 1.0
    assert selected["dirty"].values[0]
    assert not selected["removed"].values[0]

    # Nothing happens
    with pytest.warns(UserWarning):
        model.fix_variable(x[5], 1)


def test_logical_model_fix_qubo(model_qubo):
    a = model_qubo.variables("a", shape=(6,))
    model_qubo.add_interaction(a[0], coefficient=10.0, scale=1.1)
    model_qubo.add_interaction(a[1], coefficient=20.0)
    model_qubo.add_interaction(a[2], coefficient=30.0)
    model_qubo.add_interaction((a[1], a[2]), coefficient=40.0)
    model_qubo.add_interaction((a[1], a[2]), name="my interaction", coefficient=45.0)
    model_qubo.add_interaction(a[3], coefficient=50.0)
    model_qubo.add_interaction(a[4], coefficient=60.0)
    model_qubo.add_interaction((a[3], a[4]), coefficient=70.0)

    with pytest.raises(ValueError):
        model_qubo.fix_variable(target=a[0], value=-1)

    # Remove a variable that has no 2-body interactions.
    model_qubo.fix_variable(a[0], 1)
    assert model_qubo.get_fixed_size() == 1
    assert "a[0]" in model_qubo.get_fixed_array()
    assert model_qubo.get_offset() == -11.0
    selected = model_qubo.select_interaction("name == 'a[0]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]

    # Remove variables that has 2-body interactions.
    model_qubo.fix_variable(a[2], 1)
    assert model_qubo.get_fixed_size() == 2
    assert "a[2]" in model_qubo.get_fixed_array()
    assert model_qubo.get_offset() == -11.0 - 30.0
    selected = model_qubo.select_interaction("name == 'a[2]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = model_qubo.select_interaction("name == 'a[1]*a[2]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = model_qubo.select_interaction("name == 'my interaction'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = model_qubo.select_interaction("name == 'a[1] (before fixed: a[1]*a[2])'")
    assert selected["body"].values[0] == 1
    assert selected["key"].values[0] == "a[1]"
    assert selected["coefficient"].values[0] == 40.0
    assert selected["scale"].values[0] == 1.0
    assert selected["dirty"].values[0]
    assert not selected["removed"].values[0]
    selected = model_qubo.select_interaction("name == 'a[1] (before fixed: my interaction)'")
    assert selected["body"].values[0] == 1
    assert selected["key"].values[0] == "a[1]"
    assert selected["coefficient"].values[0] == 45.0
    assert selected["scale"].values[0] == 1.0
    assert selected["dirty"].values[0]
    assert not selected["removed"].values[0]

    model_qubo.fix_variable(a[4], 0)
    selected = model_qubo.select_interaction("name == 'a[4]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]
    selected = model_qubo.select_interaction("name == 'a[3]*a[4]'")
    assert selected["dirty"].values[0]
    assert selected["removed"].values[0]

    # Nothing happens
    with pytest.warns(UserWarning):
        model_qubo.fix_variable(a[5], 1)


def test_logical_model_fix_invalid_argument(model):
    with pytest.raises(TypeError):
        model.fix_variable()

    with pytest.raises(TypeError):
        model.fix_variable("invalid type")

    with pytest.raises(ValueError):
        model.fix_variable(target=None, value=1)
