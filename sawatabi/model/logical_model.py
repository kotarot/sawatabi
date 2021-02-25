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

import collections
import copy
import numbers
import pprint
import warnings

import numpy as np
import pandas as pd
import pyqubo

import sawatabi.constants as constants
from sawatabi.model.abstract_model import AbstractModel
from sawatabi.model.constraint import AbstractConstraint
from sawatabi.model.physical_model import PhysicalModel
from sawatabi.utils.functions import Functions
from sawatabi.utils.time import current_time


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
        self._default_keys = ["body", "name", "key", "key_0", "key_1", "interacts", "coefficient", "scale", "timestamp", "dirty", "removed"]
        self._interactions_array = {k: [] for k in self._default_keys}
        self._interactions_attrs = []
        self._interactions_length = 0
        self._previous_physical_model = None

    def empty(self):
        """
        Returns an empty LogicalModel with the same model type.
        """
        return LogicalModel(mtype=self._mtype)

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
        self._check_argument_type_in_tuple("shape", shape, (int, np.int64, np.int32))

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
        if fmt == constants.SELECT_DICT:
            return searched.to_dict(orient="index")
        raise ValueError(f"Format '{fmt}' is invalid.")

    def select_interactions_by_variable(self, target):
        self._update_interactions_dataframe_from_arrays()

        # Find interactions which interacts with the given variable.
        self._check_argument_type("target", target, (pyqubo.Spin, pyqubo.Binary))
        return self._interactions[(self._interactions["key_0"] == target.label) | (self._interactions["key_1"] == target.label)]["name"].values

    ################################
    # Add
    ################################

    def add_interaction(
        self,
        target,
        name="",
        coefficient=0.0,
        scale=1.0,
        attributes=None,
        timestamp=None,
    ):
        if attributes is None:
            attributes = {}
        if not target:
            raise ValueError("'target' must be specified.")
        if timestamp is None:
            timestamp = current_time()

        self._check_argument_type("coefficient", coefficient, (numbers.Number, pyqubo.core.Express, pyqubo.core.Coefficient))
        self._check_argument_type("scale", scale, (numbers.Number, pyqubo.core.Express))
        self._check_argument_type("attributes", attributes, dict)
        self._check_argument_type("timestamp", timestamp, (int, float))

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
        self._interactions_array["key_0"].append(keys[0])
        self._interactions_array["key_1"].append(keys[1])
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
        timestamp=current_time(),
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
            self._check_argument_type("timestamp", timestamp, (int, float))

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
        internal_name = self._get_internal_name_from_target_and_name(target, name)

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

    def _get_internal_name_from_target_and_name(self, target, name):
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
            raise KeyError(f"An interaction named '{internal_name}' does not exist yet in the model.")

        return internal_name

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
        self._check_argument_type("target", target, (pyqubo.Spin, pyqubo.Binary))

        # TODO: Delete variable physically
        self._deleted[target.label] = True

        # Deal with constraints
        for k, v in self.get_constraints().items():
            v.remove_variable(variables=target)

        # Remove related interations
        removed = self.select_interactions_by_variable(target)
        for r in removed:
            self.remove_interaction(name=r)

    ################################
    # Fix
    ################################

    def fix_variable(self, target, value):
        if not target:
            raise ValueError("'target' must be specified.")
        self._check_argument_type("target", target, (pyqubo.Spin, pyqubo.Binary))

        # Value check
        if self.get_mtype() == constants.MODEL_ISING:
            if value not in [1, -1]:
                raise ValueError("'value' must be one of [1, -1] because the model is an Ising model.")
        elif self.get_mtype() == constants.MODEL_QUBO:
            if value not in [1, 0]:
                raise ValueError("'value' must be one of [1, 0] because the model is a QUBO model.")

        # TODO: Delete variable physically
        self._fixed[target.label] = True

        # Update related interations
        selected = self.select_interactions_by_variable(target)

        if len(selected) <= 0:
            warnings.warn("No interactions updated.")
            return

        if value == 0:
            for s in selected:
                self.remove_interaction(name=s)
        elif value in [1, -1]:
            for s in selected:
                idx = self._interactions_array["name"].index(s)
                body = self._interactions_array["body"][idx]
                # 1-body interaction will become an offset
                if body == 1:
                    self._offset += -1 * value * self._interactions_array["coefficient"][idx] * self._interactions_array["scale"][idx]
                    self.remove_interaction(name=s)
                # 2-body interaction will become a 1-body interaction
                elif body == 2:
                    # Choose a variable that will remain
                    interacts = self._interactions_array["interacts"][idx]
                    if interacts[0].label == target.label:
                        interacts_to = interacts[1]
                    elif interacts[1].label == target.label:
                        interacts_to = interacts[0]
                    new_name = f"{interacts_to.label} (before fixed: {s})"
                    new_coefficient = value * self._interactions_array["coefficient"][idx] * self._interactions_array["scale"][idx]
                    self.add_interaction(target=interacts_to, name=new_name, coefficient=new_coefficient)
                    self.remove_interaction(name=s)

    ################################
    # Constraints
    ################################

    def add_constraint(self, constraint):
        self._check_argument_type("constraint", constraint, AbstractConstraint)
        label = constraint.get_label()
        self._constraints[label] = constraint

    def remove_constraint(self, label):
        self._check_argument_type("label", label, str)
        self._constraints.pop(label)

    ################################
    # Offset
    ################################

    def offset(self, offset):
        """
        Sets the offset value.
        """
        self._offset = offset

    def get_offset(self):
        """
        Returns the offset value.
        """
        return self._offset

    ################################
    # PyQUBO
    ################################

    def from_pyqubo(self, expression):
        self._check_argument_type("expression", expression, (pyqubo.Express, pyqubo.Model))

        if isinstance(expression, pyqubo.Express):
            pyqubo_model = expression.compile()
        else:
            pyqubo_model = expression

        compiled_qubo = pyqubo_model.compiled_qubo
        structure = pyqubo_model.structure

        for k, v in compiled_qubo.qubo.items():
            # For the first variable.
            # - structure[k[0]][0] holds the variable name,
            # - structure[k[0]][1:] holds the variable index as tuple.
            variable_0 = self.get_variables_by_name(structure[k[0]][0])
            target_0 = variable_0[structure[k[0]][1:]]

            # For the second variable.
            variable_1 = self.get_variables_by_name(structure[k[1]][0])
            target_1 = variable_1[structure[k[1]][1:]]

            coeff = v
            if isinstance(coeff, pyqubo.Coefficient):
                assert isinstance(coeff.terms, collections.defaultdict)
                for placeholder_k, _ in coeff.terms.items():
                    coeff.terms[placeholder_k] *= -1
            else:
                coeff *= -1

            if k[0] == k[1]:
                # 1-body
                self.add_interaction(target=target_0, coefficient=coeff)
            else:
                # 2-body
                self.add_interaction(target=(target_0, target_1), coefficient=coeff)

        self._offset = compiled_qubo.offset

    ################################
    # Converts
    ################################

    def to_physical(self, placeholder=None):
        if placeholder is None:
            placeholder = {}
        physical = PhysicalModel(mtype=self._mtype)

        linear, quadratic = {}, {}
        will_remove = []

        # Save the model before merging constraints to restore it later
        # original_variables = copy.deepcopy(self._variables)  # Variables will not be changed
        original_offset = self._offset
        original_deleted = copy.deepcopy(self._deleted)
        original_fixed = copy.deepcopy(self._fixed)
        # original_constraints = copy.deepcopy(self._constraints)  # Constraints will not be changed
        original_interactions_array = copy.deepcopy(self._interactions_array)
        original_interactions_attrs = copy.deepcopy(self._interactions_attrs)
        original_interactions_length = self._interactions_length

        # Resolve constraints, and convert them to the interactions
        for label, constraint in self._constraints.items():
            constraint_model = constraint.to_model()
            self.merge(constraint_model)

        # group by key
        for i in range(self._interactions_length):
            if self._interactions_array["removed"][i]:
                will_remove.append(self._interactions_array["name"][i])
                continue

            # Resolve placeholders for coefficients and scales, using PyQUBO.
            # Firstly resolve placeholders if the coefficient is already Coefficient type
            coeff_i = self._interactions_array["coefficient"][i]
            scale_i = self._interactions_array["scale"][i]
            if isinstance(coeff_i, pyqubo.core.Coefficient):
                coeff_i = coeff_i.evaluate(feed_dict=placeholder)

            # Calculate coefficient with the placeholder
            coeff_with_ph = coeff_i * scale_i
            coeff_model = (coeff_with_ph + pyqubo.Binary("sawatabi-fake-variable")).compile()  # We need a variable for a valid model for pyqubo
            coeff_ph_resolved = coeff_model.to_qubo(feed_dict=placeholder)
            coeff = coeff_ph_resolved[1]  # We don't need the variable just prepared, extracting only offset

            if self._interactions_array["body"][i] == constants.INTERACTION_LINEAR:
                if self._interactions_array["key"][i] in linear:
                    linear[self._interactions_array["key"][i]] += coeff
                else:
                    linear[self._interactions_array["key"][i]] = coeff

            elif self._interactions_array["body"][i] == constants.INTERACTION_QUADRATIC:
                if self._interactions_array["key"][i] in quadratic:
                    quadratic[self._interactions_array["key"][i]] += coeff
                else:
                    quadratic[self._interactions_array["key"][i]] = coeff

        # For offset as well
        offset = self._offset
        if isinstance(self._offset, pyqubo.core.Coefficient):
            offset = self._offset.evaluate(feed_dict=placeholder)
        # Calculate coefficient with the placeholder
        offset_model = (offset + pyqubo.Binary("sawatabi-fake-variable")).compile()  # We need a variable for a valid model for pyqubo
        offset_ph_resolved = offset_model.to_qubo(feed_dict=placeholder)
        offset = offset_ph_resolved[1]  # We don't need the variable just prepared, extracting only offset

        # set to physical
        for k, v in linear.items():
            if v != 0.0:
                physical.add_interaction(k, body=constants.INTERACTION_LINEAR, coefficient=v)
                physical._variables_set.add(k)
        for k, v in quadratic.items():
            if v != 0.0:
                physical.add_interaction(k, body=constants.INTERACTION_QUADRATIC, coefficient=v)
                physical._variables_set.add(k[0])
                physical._variables_set.add(k[1])
        physical._offset = offset

        # label_to_index / index_to_label
        current_index = 0
        for val in self._variables.values():
            flattened = list(Functions._flatten(val.bit_list))
            for v in flattened:
                if (v.label not in self._deleted) and (v.label in physical._variables_set):
                    physical._label_to_index[v.label] = current_index
                    physical._index_to_label[current_index] = v.label
                    current_index += 1

        # save the last physical model
        self._previous_physical_model = physical

        # Restore the model before adding constraints
        # self._variables = original_variables  # Variables were not changed
        self._offset = original_offset
        self._deleted = original_deleted
        self._fixed = original_fixed
        # self._constraints = original_constraints  # Constraints were not changed
        self._interactions_array = original_interactions_array
        self._interactions_attrs = original_interactions_attrs
        self._interactions_length = original_interactions_length

        # Remove interactions
        # TODO: Physically remove the logically removed interactions
        for rm in will_remove:
            idx = self._interactions_array["name"].index(rm)
            for k in self._interactions_array.keys():
                self._interactions_array[k].pop(idx)
            self._interactions_length -= 1

        # Set dirty flag
        for i in range(self._interactions_length):
            # TODO: Calc difference from previous physical model by referencing dirty flags.
            if self._interactions_array["dirty"][i]:
                self._interactions_array["dirty"][i] = False

        return physical

    def merge(self, other):
        self._check_argument_type("other", other, LogicalModel)

        # Check type
        if self._mtype != other._mtype:
            other._convert_mtype()

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
        if self._mtype == constants.MODEL_QUBO:
            self.to_ising()
        elif self._mtype == constants.MODEL_ISING:
            self.to_qubo()

    def _update_variables_type(self):
        vartype = self._modeltype_to_vartype(self._mtype)
        for name, variable in self._variables.items():
            self._variables[name] = pyqubo.Array.create(name, shape=variable.shape, vartype=vartype)
        df = self.select_interaction(query="removed == False")
        for index, interaction in df.iterrows():
            interacts = interaction["interacts"]
            if self._mtype == constants.MODEL_ISING:
                if isinstance(interacts, tuple):
                    self._interactions_array["interacts"][index] = (
                        pyqubo.Spin(interaction["interacts"][0].label),
                        pyqubo.Spin(interaction["interacts"][1].label),
                    )
                else:
                    self._interactions_array["interacts"][index] = pyqubo.Spin(interaction["interacts"].label)
            elif self._mtype == constants.MODEL_QUBO:
                if isinstance(interacts, tuple):
                    self._interactions_array["interacts"][index] = (
                        pyqubo.Binary(interaction["interacts"][0].label),
                        pyqubo.Binary(interaction["interacts"][1].label),
                    )
                else:
                    self._interactions_array["interacts"][index] = pyqubo.Binary(interaction["interacts"].label)

    def to_ising(self):
        """
        For h:
            hx = h*(s+1)/2 = hs/2+h/2
        For J:
            Jxy = J*(s+1)/2*(t+1)/2 = Jst/4+Js/4+Jt/4+J/4
        """
        if self._mtype != constants.MODEL_ISING:
            self._mtype = constants.MODEL_ISING

            # Update variables from Spin to Binary
            self._update_variables_type()

            # Update h_{i}
            h_df = self.select_interaction(query="(body == 1) and (removed == False)")
            for index, h in h_df.iterrows():
                coeff = h["coefficient"]
                self.update_interaction(name=h["name"], coefficient=coeff * 0.5)
                self._offset += coeff * 0.5

            # Update J_{ij}
            J_df = self.select_interaction(query="(body == 2) and (removed == False)")
            for _, J in J_df.iterrows():
                coeff = J["coefficient"]
                self.update_interaction(name=J["name"], coefficient=coeff * 0.25)
                self.add_interaction(
                    target=J["interacts"][0], name=f"{J['key_0']} from {J['name']} (mtype additional {current_time()})", coefficient=coeff * 0.25
                )
                self.add_interaction(
                    target=J["interacts"][1], name=f"{J['key_1']} from {J['name']} (mtype additional {current_time()})", coefficient=coeff * 0.25
                )
                self._offset += coeff * 0.25
        else:
            warnings.warn("The model is already an Ising model.")

    def to_qubo(self):
        """
        For h:
            hs = h*(2x-1) = 2hx-h
        For J:
            Jst = J*(2x-1)*(2y-1) = 4Jxy_2Jx-2Jy+J
        """
        if self._mtype != constants.MODEL_QUBO:
            self._mtype = constants.MODEL_QUBO

            # Update variables from Spin to Binary
            self._update_variables_type()

            # Update h_{i}
            h_df = self.select_interaction(query="(body == 1) and (removed == False)")
            for index, h in h_df.iterrows():
                coeff = h["coefficient"]
                self.update_interaction(name=h["name"], coefficient=coeff * 2.0)
                self._offset -= coeff

            # Update J_{ij}
            J_df = self.select_interaction(query="(body == 2) and (removed == False)")
            for _, J in J_df.iterrows():
                coeff = J["coefficient"]
                self.update_interaction(name=J["name"], coefficient=J["coefficient"] * 4.0)
                self.add_interaction(
                    target=J["interacts"][0], name=f"{J['key_0']} from {J['name']} (mtype additional {current_time()})", coefficient=-coeff * 2.0
                )
                self.add_interaction(
                    target=J["interacts"][1], name=f"{J['key_1']} from {J['name']} (mtype additional {current_time()})", coefficient=-coeff * 2.0
                )
                self._offset += coeff
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

    def get_attributes(self, target=None, name=""):
        """
        Returns a dict of attributes (keys and values) for the given variable or interaction.
        """
        internal_name = self._get_internal_name_from_target_and_name(target, name)
        idx = self._interactions_array["name"].index(internal_name)
        res = {}
        for attr in self._interactions_attrs:
            res[attr] = self._interactions_array[attr][idx]
        return res

    def get_attribute(self, target=None, name="", key=""):
        """
        Returns the value of the key for the given variable or interaction.
        """
        self._check_argument_type("key", key, str)
        attributes = self.get_attributes(target, name)
        return attributes[key]

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

    def __eq__(self, other):
        return (
            isinstance(other, LogicalModel)
            and (self._mtype == other._mtype)
            and (self._variables == other._variables)
            and (self._interactions_array == other._interactions_array)
            and (self._interactions_attrs == other._interactions_attrs)
            and (self._interactions_length == other._interactions_length)
            and (self._constraints == other._constraints)
            and (self._deleted == other._deleted)
            and (self._fixed == other._fixed)
            and (self._offset == other._offset)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        self._update_interactions_dataframe_from_arrays()

        s = "LogicalModel({"
        s += "'mtype': '" + str(self._mtype) + "', "
        s += "'variables': " + self.remove_leading_spaces(str(self._variables)) + ", "
        if self._interactions.empty:
            s += "'interactions': 'Empty', "
        else:
            s += "'interactions': " + self._interactions.to_json(orient="values") + ", "
        s += "'offset': " + str(self._offset) + ", "
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
        for name, variables in self._variables.items():
            s.append("┃  name: " + name)
            s.append(self.append_prefix(str(variables), length=4))
        s.append("┣━ interactions:")
        if self._interactions.empty:
            s.append(self.append_prefix("Empty", length=4))
        else:
            s.append(self.append_prefix(pprint.pformat(self._interactions), length=4))
        s.append("┣━ offset: " + str(self._offset))
        s.append("┣━ constraints:")
        s.append(self.append_prefix(pprint.pformat(self._constraints), length=4))
        s.append("┗" + ("━" * 64))
        return "\n".join(s)
