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

import numbers
import pprint
import warnings

import numpy as np
import pandas as pd
import pyqubo

import sawatabi.constants as constants
from sawatabi.model.abstract_model import AbstractModel
from sawatabi.model.dependency_constraint import DependencyConstraint
from sawatabi.model.n_hot_constraint import NHotConstraint
from sawatabi.model.physical_model import PhysicalModel
from sawatabi.utils.functions import Functions
from sawatabi.utils.time import current_time_ms


class LogicalModel(AbstractModel):
    def __init__(self, mtype=""):
        super().__init__(mtype)

        # Note: Cannot rename to 'variables' because we already have 'variables' method.
        self._variables = {}
        self._offset = 0.0
        self._deleted = {}
        self._fixed = {}
        self._constraints = {}
        self._interactions = None
        self._default_keys = ["body", "name", "key", "key.0", "key.1", "interacts", "coefficient", "scale", "timestamp", "dirty", "removed"]
        self._interactions_array = {k: [] for k in self._default_keys}
        self._interactions_attrs = []
        self._interactions_length = 0
        self._previous_physical_model = None

    ################################
    # Variables
    ################################

    def variables(self, name, shape=()):
        if isinstance(name, pyqubo.Array):
            flattened = list(Functions._flatten(name.bit_list))
            if ((self._mtype == constants.MODEL_ISING) and isinstance(flattened[0], pyqubo.Binary)) or (
                (self._mtype == constants.MODEL_QUBO) and isinstance(flattened[0], pyqubo.Spin)
            ):
                raise TypeError("Model type and PyQUBO Array type mismatch.")

            # Retrieve label from the pyqubo variable
            found = flattened[0].label.index("[")
            this_name = flattened[0].label[:found]

            self._variables[this_name] = name
            return self._variables[this_name]

        self._check_argument_type("name", name, str)
        self._check_argument_type("shape", shape, tuple)
        self._check_argument_type_in_tuple("shape", shape, int)

        vartype = self._modeltype_to_vartype(self._mtype)

        self._variables[name] = pyqubo.Array.create(name, shape=shape, vartype=vartype)
        return self._variables[name]

    def append(self, name, shape=()):
        self._check_argument_type("name", name, str)
        self._check_argument_type("shape", shape, tuple)
        self._check_argument_type_in_tuple("shape", shape, (int, np.int64))

        if name not in self._variables:
            # raise KeyError(f"Variables name '{name}' is not defined in the model.")
            warnings.warn(f"Variables name '{name}' is not defined in the model, but will be created instead of appending it.")
            return self.variables(name, shape)

        # tuple elementwise addition
        new_shape = Functions.elementwise_add(self._variables[name].shape, shape)
        vartype = self._modeltype_to_vartype(self._mtype)

        self._variables[name] = pyqubo.Array.create(name, shape=new_shape, vartype=vartype)
        return self._variables[name]

    ################################
    # Select
    ################################

    def select_interaction(self, query, fmt=constants.SELECT_SERIES):
        self._update_interactions_dataframe_from_arrays()

        searched = self._interactions.query(query)

        if fmt == constants.SELECT_SERIES:
            return searched
        elif fmt == constants.SELECT_DICT:
            return searched.to_dict(orient="index")
        else:
            raise ValueError(f"Format '{fmt}' is invalid.")

    def select_interactions_by_variable(self, target):
        self._update_interactions_dataframe_from_arrays()

        # Find interactions which interacts with the given variable.
        self._check_argument_type("target", target, (pyqubo.Spin, pyqubo.Binary))
        return self._interactions[(self._interactions["key.0"] == target.label) | (self._interactions["key.1"] == target.label)]["name"].values

    ################################
    # Add
    ################################

    def add_interaction(
        self,
        target,
        name="",
        coefficient=0.0,
        scale=1.0,
        attributes={},
        timestamp=current_time_ms(),
    ):
        if not target:
            raise ValueError("'target' must be specified.")

        self._check_argument_type("coefficient", coefficient, numbers.Number)
        self._check_argument_type("scale", scale, numbers.Number)
        self._check_argument_type("attributes", attributes, dict)
        self._check_argument_type("timestamp", timestamp, int)

        interaction_info = self._get_interaction_info_from_target(target)

        body = interaction_info["body"]
        if name:
            # Use the given specific name
            self._check_argument_type("name", name, str)
            internal_name = name
        else:
            # Automatically named by the default name
            internal_name = interaction_info["name"]

        if self._has_name(internal_name):
            if not self._is_removed(internal_name):
                raise ValueError(f"An interaction named '{internal_name}' already exists. Cannot add the same name.")
            else:
                raise ValueError(f"An interaction named '{internal_name}' is already removed.")

        if body == 1:
            keys = (interaction_info["key"], np.nan)
        elif body == 2:
            keys = interaction_info["key"]

        # Adding a dict to Pandas DataFrame is slow.
        # We need to expand the internal arrays and generate a DataFrame based on them.
        self._interactions_array["body"].append(body)
        self._interactions_array["name"].append(internal_name)
        self._interactions_array["key"].append(interaction_info["key"])
        self._interactions_array["key.0"].append(keys[0])
        self._interactions_array["key.1"].append(keys[1])
        self._interactions_array["interacts"].append(interaction_info["interacts"])
        self._interactions_array["coefficient"].append(coefficient)
        self._interactions_array["scale"].append(scale)
        self._interactions_array["timestamp"].append(timestamp)
        # Note: dirty flag (= modification flag) means this interaction has not converted to a physical model yet.
        # dirty flag will be False when the interaction is written to a physical model.
        self._interactions_array["dirty"].append(True)
        self._interactions_array["removed"].append(False)

        # Expand existing attributes
        for attr in self._interactions_attrs:
            self._interactions_array[attr].append(np.nan)

        # Set new attributes
        for k, v in attributes.items():
            attrs_key = f"attributes.{k}"
            if attrs_key not in self._interactions_array:
                self._interactions_array[attrs_key] = [np.nan] * (self._interactions_length + 1)
                self._interactions_attrs.append(attrs_key)
            self._interactions_array[attrs_key][-1] = v

        self._interactions_length += 1

    ################################
    # Update
    ################################

    def update_interaction(
        self,
        target=None,
        name="",
        coefficient=None,
        scale=None,
        attributes=None,
        timestamp=current_time_ms(),
    ):
        if (not target) and (not name):
            raise ValueError("Either 'target' or 'name' must be specified.")
        if target and name:
            raise ValueError("Both 'target' and 'name' cannot be specified simultaneously.")

        if coefficient is not None:
            self._check_argument_type("coefficient", coefficient, numbers.Number)
        if scale is not None:
            self._check_argument_type("scale", scale, numbers.Number)
        if attributes is not None:
            self._check_argument_type("attributes", attributes, dict)
        if timestamp is not None:
            self._check_argument_type("timestamp", timestamp, int)

        if target is not None:
            interaction_info = self._get_interaction_info_from_target(target)

        if name:
            internal_name = name
            if not self._has_name(internal_name):
                raise KeyError(f"An interaction named '{internal_name}' does not exist yet in the model. Need to be added before updating.")
            # Already given the specific name
            self._check_argument_type("name", name, (str, tuple))
        else:
            # Will be automatically named by the default name
            internal_name = interaction_info["name"]
            if not self._has_name(internal_name):
                warnings.warn(f"An interaction named '{internal_name}' does not exist yet in the model, but will be added instead of updating it.")
                _coefficient = 0.0 if coefficient is None else coefficient
                _scale = 1.0 if scale is None else scale
                return self.add_interaction(target=target, coefficient=_coefficient, scale=_scale, timestamp=timestamp)

        if self._is_removed(internal_name):
            raise ValueError(f"An interaction named '{internal_name}' is already removed.")

        update_idx = self._interactions_array["name"].index(internal_name)

        # update only properties which is given by arguments
        if coefficient is not None:
            self._interactions_array["coefficient"][update_idx] = coefficient
        if scale is not None:
            self._interactions_array["scale"][update_idx] = scale
        if attributes is not None:
            for k, v in attributes.items():
                attrs_key = f"attributes.{k}"
                if attrs_key not in self._interactions_array:
                    self._interactions_array[attrs_key] = [np.nan] * self._interactions_length
                self._interactions_array[attrs_key][update_idx] = v
        self._interactions_array["timestamp"][update_idx] = timestamp
        self._interactions_array["dirty"][update_idx] = True

    ################################
    # Remove
    ################################

    def remove_interaction(self, target=None, name=""):
        if (not target) and (not name):
            raise ValueError("Either 'target' or 'name' must be specified.")
        if target and name:
            raise ValueError("Both 'target' and 'name' cannot be specified simultaneously.")

        if target is not None:
            interaction_info = self._get_interaction_info_from_target(target)

        if name:
            # Already given the specific name
            self._check_argument_type("name", name, (str, tuple))
            internal_name = name
        else:
            # Will be automatically named by the default name
            internal_name = interaction_info["name"]

        if not self._has_name(internal_name):
            raise KeyError(f"An interaction named '{internal_name}' does not exist yet. Need to be added before updating.")

        # Don't need to check this. Removing will be overwritten.
        # if self._is_removed(internal_name):
        #     raise ValueError(f"An interaction named '{internal_name}' is already removed.")

        remove_idx = self._interactions_array["name"].index(internal_name)

        # logically remove
        # This will be physically removed when it's converted to a physical model.
        self._interactions_array["removed"][remove_idx] = True
        self._interactions_array["dirty"][remove_idx] = True

    ################################
    # Helper methods for add, update, remove, and select
    ################################

    @staticmethod
    def _get_interaction_info_from_target(target):
        if isinstance(target, (pyqubo.Spin, pyqubo.Binary)):
            body = constants.INTERACTION_LINEAR
            interacts = target
            key = target.label
            name = target.label
        elif isinstance(target, tuple):
            if len(target) != 2:
                raise TypeError("The length of a tuple 'target' must be two.")
            for i in target:
                if not isinstance(i, (pyqubo.Spin, pyqubo.Binary)):
                    raise TypeError("All elements of 'target' must be a 'pyqubo.Spin' or 'pyqubo.Binary'.")
            body = constants.INTERACTION_QUADRATIC

            if target[0].label == target[1].label:
                raise ValueError("The given target is not a valid interaction.")

            # Tuple elements to dictionary order
            if target[0].label < target[1].label:
                interacts = (target[0], target[1])
                key = (target[0].label, target[1].label)
                name = f"{target[0].label}*{target[1].label}"
            else:
                interacts = (target[1], target[0])
                key = (target[1].label, target[0].label)
                name = f"{target[1].label}*{target[0].label}"
        else:
            raise TypeError("Invalid 'target'.")
        return {"body": body, "interacts": interacts, "key": key, "name": name}

    def _has_name(self, internal_name):
        return internal_name in self._interactions_array["name"]

    def _is_removed(self, internal_name):
        idx = self._interactions_array["name"].index(internal_name)
        return self._interactions_array["removed"][idx]

    def _update_interactions_dataframe_from_arrays(self):
        # Generate a DataFrame from the internal interaction arrays.
        # If we create new DataFrame every interaction update, computation time consumes a lot.
        # We only generate a DataFrame just before we need it.
        self._interactions = pd.DataFrame(self._interactions_array)

    ################################
    # Delete
    ################################

    def delete_variable(self, target):
        if not target:
            raise ValueError("'target' must be specified.")
        self._check_argument_type("target", target, (pyqubo.Array, pyqubo.Spin, pyqubo.Binary))

        # TODO: Delete variable physically
        self._deleted[target.label] = True

        # Deal with constraints
        for k, v in self.get_constraints().items():
            if v._constraint_type == "NHotConstraint":
                self.n_hot_constraint(target, n=v._n, strength=v._strength, label=v._label, delete_flag=True)

        # Remove related interations
        removed = self.select_interactions_by_variable(target)
        for r in removed:
            self.remove_interaction(name=r)

    ################################
    # Fix
    ################################

    def fix_variable(self, target, value):
        raise NotImplementedError

    ################################
    # PyQUBO
    ################################

    def from_pyqubo(self, expression):
        if not (isinstance(expression, pyqubo.Express) or isinstance(expression, pyqubo.Model)):
            raise TypeError("'expression' must be a PyQUBO Expression (pyqubo.Express) or a PyQUBO Model (pyqubo.Model).")
        raise NotImplementedError

    ################################
    # Constraints
    ################################

    def n_hot_constraint(self, target, n=1, strength=1.0, label=constants.DEFAULT_LABEL_N_HOT, delete_flag=False):
        self._check_argument_type("target", target, (pyqubo.Array, pyqubo.Spin, pyqubo.Binary, list))
        self._check_argument_type("n", n, int)
        self._check_argument_type("strength", strength, numbers.Number)
        self._check_argument_type("label", label, str)

        if isinstance(target, list):
            self._check_argument_type_in_list("target", target, (pyqubo.Spin, pyqubo.Binary))
        if not (isinstance(target, pyqubo.Array) or isinstance(target, list)):
            target = [target]
        if label not in self._constraints:
            self._constraints[label] = NHotConstraint(n, strength, label)
        original_variables = self._constraints[label]._variables
        added_variables = set()
        for t in target:
            added_variables = added_variables.union(set([t]))
            self._constraints[label].add(set([t]))

        # If delete_flag is set, the target variable will be deleted from the constraint.
        if delete_flag:
            remaining_variables = original_variables - added_variables
            self._constraints[label]._variables -= added_variables
            if self._mtype == constants.MODEL_QUBO:
                # Do nothing for QUBO model (only remove adjacent interactions)
                pass
            elif self._mtype == constants.MODEL_ISING:
                # For Ising model, all linear interactions of variables for the constraint are taken care.
                num_variables = len(remaining_variables)
                for rvar in remaining_variables:
                    coeff = -1.0 * strength * (num_variables - 2 * n)
                    self.update_interaction(name=f"{rvar.label} ({label})", coefficient=coeff)

        else:
            additional_variables = added_variables - original_variables
            if self._mtype == constants.MODEL_QUBO:
                # For QUBO model, only interactions of additinal variables are taken care.
                for avar in additional_variables:
                    coeff = -1.0 * strength * (1 - 2 * n)
                    self.add_interaction(avar, name=f"{avar.label} ({label})", coefficient=coeff)
                    for adj in additional_variables:
                        if avar.label < adj.label:
                            coeff = -2.0 * strength
                            self.add_interaction((avar, adj), name=f"{avar.label}*{adj.label} ({label})", coefficient=coeff)
                    for adj in original_variables:
                        coeff = -2.0 * strength
                        self.add_interaction((avar, adj), name=f"{avar.label}*{adj.label} ({label})", coefficient=coeff)

            elif self._mtype == constants.MODEL_ISING:
                # For Ising model, all interactions of variables for the constraint are taken care.
                num_variables = len(original_variables) + len(additional_variables)
                for ovar in original_variables:
                    coeff = -1.0 * strength * (num_variables - 2 * n)
                    self.update_interaction(name=f"{ovar.label} ({label})", coefficient=coeff)
                for avar in additional_variables:
                    coeff = -1.0 * strength * (num_variables - 2 * n)
                    self.add_interaction(avar, name=f"{avar.label} ({label})", coefficient=coeff)
                    for adj in additional_variables:
                        if avar.label < adj.label:
                            coeff = -1.0 * strength
                            self.add_interaction((avar, adj), name=f"{avar.label}*{adj.label} ({label})", coefficient=coeff)
                    for adj in original_variables:
                        coeff = -1.0 * strength
                        self.add_interaction((avar, adj), name=f"{avar.label}*{adj.label} ({label})", coefficient=coeff)

    def dependency_constraint(self, target_src, target_dst, strength=1.0, label=constants.DEFAULT_LABEL_DEPENDENCY):
        self._check_argument_type("strength", strength, numbers.Number)
        self._check_argument_type("label", label, str)
        _ = DependencyConstraint()
        raise NotImplementedError

    ################################
    # Converts
    ################################

    def to_physical(self, placeholder={}):
        # TODO:
        # - resolve placeholder

        physical = PhysicalModel(mtype=self._mtype)

        linear, quadratic = {}, {}
        will_remove = []

        # label_to_index / index_to_label
        current_index = 0
        for val in self._variables.values():
            flattened = list(Functions._flatten(val.bit_list))
            for v in flattened:
                if v.label not in self._deleted:
                    physical._label_to_index[v.label] = current_index
                    physical._index_to_label[current_index] = v.label
                    current_index += 1

        # group by key
        for i in range(self._interactions_length):
            if self._interactions_array["removed"][i]:
                will_remove.append(self._interactions_array["name"][i])
                continue

            # TODO: Calc difference from previous physical model by referencing dirty flags.
            if self._interactions_array["dirty"][i]:
                self._interactions_array["dirty"][i] = False

            if self._interactions_array["body"][i] == constants.INTERACTION_LINEAR:
                if self._interactions_array["key"][i] in linear:
                    linear[self._interactions_array["key"][i]] += float(self._interactions_array["coefficient"][i] * self._interactions_array["scale"][i])
                else:
                    linear[self._interactions_array["key"][i]] = float(self._interactions_array["coefficient"][i] * self._interactions_array["scale"][i])

            elif self._interactions_array["body"][i] == constants.INTERACTION_QUADRATIC:
                if self._interactions_array["key"][i] in quadratic:
                    quadratic[self._interactions_array["key"][i]] += float(self._interactions_array["coefficient"][i] * self._interactions_array["scale"][i])
                else:
                    quadratic[self._interactions_array["key"][i]] = float(self._interactions_array["coefficient"][i] * self._interactions_array["scale"][i])

        # TODO: Physically remove the logically removed interactions
        for rm in will_remove:
            idx = self._interactions_array["name"].index(rm)
            for k in self._interactions_array.keys():
                self._interactions_array[k].pop(idx)
            self._interactions_length -= 1

        # set to physical
        for k, v in linear.items():
            physical.add_interaction(k, body=constants.INTERACTION_LINEAR, coefficient=v)
        for k, v in quadratic.items():
            physical.add_interaction(k, body=constants.INTERACTION_QUADRATIC, coefficient=v)

        # save the last physical model
        self._previous_physical_model = physical

        return physical

    def merge(self, other):
        self._check_argument_type("other", other, LogicalModel)

        # Check type
        if self._mtype != other._mtype:
            # Currently...
            raise ValueError("Cannot merge different model type.")

        # Check variables
        for key, value in self._variables.items():
            if key in other._variables.keys():
                if len(value.shape) != len(other._variables[key].shape):
                    raise ValueError(f"Cannot merge model since the dimension of '{key}' is different.")

        # Merge variables
        for key, value in other._variables.items():
            if key not in self._variables:
                self._variables[key] = value
            else:
                shape_current = self._variables[key].shape
                shape_max = Functions.elementwise_max(value.shape, shape_current)
                shape_diff = Functions.elementwise_sub(shape_max, shape_current)
                self.append(name=key, shape=shape_diff)

        # Merge interactions
        merged_interactions_with_duplication = {k: [] for k in self._default_keys}
        merged_attrs = list(set(self._interactions_attrs) | set(other._interactions_attrs))
        for k in self._default_keys:
            merged_interactions_with_duplication[k] = self._interactions_array[k] + other._interactions_array[k]
        for attr in merged_attrs:
            if (attr in self._interactions_attrs) and (attr in other._interactions_array):
                merged_interactions_with_duplication[attr] = self._interactions_array[attr] + other._interactions_array[attr]
            elif attr in self._interactions_attrs:
                merged_interactions_with_duplication[attr] = self._interactions_array[attr] + ([np.nan] * other._interactions_length)
            elif attr in other._interactions_attrs:
                merged_interactions_with_duplication[attr] = ([np.nan] * self._interactions_length) + other._interactions_array[attr]
        duplicate_names = list(set(self._interactions_array["name"]) & set(other._interactions_array["name"]))

        # Rename duplicate interaction names by adding suffix of model id
        for idx, name in enumerate(merged_interactions_with_duplication["name"]):
            if name in duplicate_names:
                if idx < self._interactions_length:
                    model_id = id(self)
                else:
                    model_id = id(other)
                merged_interactions_with_duplication["name"][idx] = f"{name} ({model_id})"

        self._interactions_array = merged_interactions_with_duplication
        self._interactions_attrs = merged_attrs
        self._interactions_length = self._interactions_length + other._interactions_length

        # Merge constraints
        # If both models have a constraint with the same label, cannnot merge currently
        if len(set(self._constraints.keys()) & set(other._constraints.keys())) > 0:
            raise ValueError("Cannot merge model since both model have a constraint with the same label.")
        self._constraints.update(other._constraints)

        # Merge other fields
        self._offset += other._offset
        self._deleted.update(other._deleted)
        self._fixed.update(other._fixed)

    def _convert_mtype(self):
        """
        Converts the model to a QUBO model if the current model type is Ising, and vice versa.
        """
        raise NotImplementedError

    def to_ising(self):
        if self._mtype != constants.MODEL_ISING:
            self._convert_mtype()
        else:
            warnings.warn("The model is already an Ising model.")

    def to_qubo(self):
        if self._mtype != constants.MODEL_QUBO:
            self._convert_mtype()
        else:
            warnings.warn("The model is already a QUBO model.")

    ################################
    # Getters
    ################################

    def get_variables(self):
        """
        Returns a list of alive variables (i.e., variables which are not removed nor fixed).
        """
        return self._variables

    def get_variables_by_name(self, name):
        """
        Returns a list of alive variables (i.e., variables which are not removed nor fixed) by the given name.
        """
        return self._variables[name]

    def get_deleted_array(self):
        """
        Returns a list of variables which are deleted.
        """
        return list(self._deleted.keys())

    def get_fixed_array(self):
        """
        Returns a list of variables which are fixed.
        """
        return list(self._fixed.keys())

    def get_size(self):
        """
        Returns the number of all alive variables (i.e., variables which are not removed or fixed).
        """
        return self.get_all_size() - self.get_deleted_size() - self.get_fixed_size()

    def get_deleted_size(self):
        """
        Returns the number of variables which are deleted.
        """
        return len(self._deleted)

    def get_fixed_size(self):
        """
        Returns the number of variables which are fixed.
        """
        return len(self._fixed)

    def get_all_size(self):
        """
        Return the number of all variables including removed or fixed.
        """
        all_size = 0
        for _, variables in self.get_variables().items():
            size = 1
            for s in variables.shape:
                size *= s
            all_size += size
        return all_size

    def get_attributes(self, target):
        """
        Returns a dict of attributes (keys and values) for the given variable or interaction.
        """
        raise NotImplementedError

    def get_attribute(self, target, key):
        """
        Returns the value of the key for the given variable or interaction.
        """
        raise NotImplementedError

    def get_constraints(self):
        """
        Returns a list of constraints.
        """
        return self._constraints

    def get_constraints_by_label(self, label):
        """
        Returns a list of constraints by the given label.
        """
        return self._constraints[label]

    ################################
    # Built-in functions
    ################################

    def __repr__(self):
        self._update_interactions_dataframe_from_arrays()

        s = "LogicalModel({"
        s += "'mtype': '" + str(self._mtype) + "', "
        s += "'variables': " + self.remove_leading_spaces(str(self._variables)) + ", "
        s += "'interactions': " + str(self._interactions) + ", "
        s += "'constraints': " + str(self._constraints) + "})"
        return s

    def __str__(self):
        self._update_interactions_dataframe_from_arrays()

        s = []
        s.append("┏" + ("━" * 64))
        s.append("┃ LOGICAL MODEL")
        s.append("┣" + ("━" * 64))
        s.append("┣━ mtype: " + str(self._mtype))
        s.append("┣━ variables: " + str(list(self._variables.keys())))
        for name, vars in self._variables.items():
            s.append("┃  name: " + name)
            s.append(self.append_prefix(str(vars), length=4))
        s.append("┣━ interactions:")
        s.append(self.append_prefix(pprint.pformat(self._interactions), length=4))
        s.append("┣━ constraints:")
        s.append(self.append_prefix(pprint.pformat(self._constraints), length=4))
        s.append("┗" + ("━" * 64))
        return "\n".join(s)
