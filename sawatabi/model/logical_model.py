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

import numbers
import pprint
import warnings

import pyqubo

import sawatabi.constants as constants
from sawatabi.model.abstract_model import AbstractModel
from sawatabi.model.n_hot_constraint import NHotConstraint
from sawatabi.model.physical_model import PhysicalModel
from sawatabi.utils.functions import Functions
from sawatabi.utils.time import current_time_ms


class LogicalModel(AbstractModel):
    def __init__(self, mtype=""):
        super().__init__(mtype)
        self._constraints = {}
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
        self._check_argument_type_in_tuple("shape", shape, int)

        if name not in self._variables:
            # raise KeyError(f"Variables name '{name}' is not defined in the model.")
            warnings.warn(
                f"Variables name '{name}' is not defined in the model, but will be created instead of appending it."
            )
            return self.variables(name, shape)

        # tuple elementwise addition
        new_shape = Functions.elementwise_add(self._variables[name].shape, shape)
        vartype = self._modeltype_to_vartype(self._mtype)

        self._variables[name] = pyqubo.Array.create(name, shape=new_shape, vartype=vartype)
        return self._variables[name]

    ################################
    # Select
    ################################

    def select_variable(self):
        raise NotImplementedError

    def select_interaction(self):
        raise NotImplementedError

    def select_interaction_tmp(self, target):
        # TODO: This is a temporal implementation.
        # Refactor this using pandas later.
        selected = []
        for k, v in self._interactions[constants.INTERACTION_LINEAR].items():
            if v["interacts"] == target:
                selected.append(v["name"])
        for k, v in self._interactions[constants.INTERACTION_QUADRATIC].items():
            if (v["interacts"][0] == target) or (v["interacts"][1] == target):
                selected.append(v["name"])
        return selected

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

        if internal_name in self._interactions[body]:
            if not self._interactions[body][internal_name]["removed"]:
                raise ValueError(f"An interaction named '{internal_name}' already exists. Cannot add the same name.")
            else:
                raise ValueError(f"An interaction named '{internal_name}' is already removed.")

        # Note: dirty flag (= modification flag) means this interaction has not converted to a physical model yet.
        # dirty flag will be False when the interaction is written to a physical model.
        add_object = {
            "name": internal_name,
            "key": interaction_info["key"],
            "interacts": interaction_info["interacts"],
            "coefficient": coefficient,
            "scale": scale,
            "attributes": attributes,
            "timestamp": timestamp,
            "dirty": True,
            "removed": False,
        }
        self._interactions[body][internal_name] = add_object
        return add_object

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

        body = None
        if name:
            # Already given the specific name
            self._check_argument_type("name", name, (str, tuple))
            internal_name = name
            for b in [constants.INTERACTION_LINEAR, constants.INTERACTION_QUADRATIC]:
                if name in self._interactions[b]:
                    body = b
                    break
        else:
            # Will be automatically named by the default name
            body = interaction_info["body"]
            internal_name = interaction_info["name"]

        if (body is None) or (internal_name not in self._interactions[body]):
            if target is not None:
                warnings.warn(
                    f"An interaction named '{internal_name}' does not exist yet in the model,"
                    " but will be added instead of updating it."
                )
                _coefficient = 0.0 if coefficient is None else coefficient
                _scale = 1.0 if scale is None else scale
                return self.add_interaction(target=target, coefficient=_coefficient, scale=_scale, timestamp=timestamp)
            else:
                raise KeyError(
                    f"An interaction named '{internal_name}' does not exist yet in the model."
                    " Need to be added before updating."
                )
        if self._interactions[body][internal_name]["removed"]:
            raise ValueError(f"An interaction named '{internal_name}' is already removed.")

        # update only properties which is given by arguments
        if coefficient is not None:
            self._interactions[body][internal_name]["coefficient"] = coefficient
        if scale is not None:
            self._interactions[body][internal_name]["scale"] = scale
        if attributes is not None:
            self._interactions[body][internal_name]["attributes"] = attributes
        self._interactions[body][internal_name]["timestamp"] = timestamp
        self._interactions[body][internal_name]["dirty"] = True
        assert not self._interactions[body][internal_name]["removed"]

        return self._interactions[body][internal_name]

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

        body = None
        if name:
            # Already given the specific name
            self._check_argument_type("name", name, (str, tuple))
            internal_name = name
            for b in [constants.INTERACTION_LINEAR, constants.INTERACTION_QUADRATIC]:
                if name in self._interactions[b]:
                    body = b
                    break
        else:
            # Will be automatically named by the default name
            body = interaction_info["body"]
            internal_name = interaction_info["name"]

        if (body is None) or (internal_name not in self._interactions[body]):
            raise KeyError(
                f"An interaction named '{internal_name}' does not exist yet. Need to be added before updating."
            )

        # logically remove
        # This will be physically removed when it's converted to a physical model.
        self._interactions[body][internal_name]["removed"] = True
        self._interactions[body][internal_name]["dirty"] = True

        return self._interactions[body][internal_name]

    ################################
    # Helper methods for add, update, and remove
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

    ################################
    # Delete
    ################################

    def delete_variable(self, target):
        if not target:
            raise ValueError("'target' must be specified.")
        self._check_argument_type("target", target, (pyqubo.Array, pyqubo.Spin, pyqubo.Binary))

        # TODO: Delete variable physically
        self._deleted.append(target)

        # Deal with constraints
        for k, v in self.get_constraints().items():
            if v._constraint_type == "NHotConstraint":
                self.n_hot_constraint(target, n=v._n, strength=v._strength, label=v._label, delete_flag=True)

        # Remove related interations
        removed = self.select_interaction_tmp(target)
        for r in removed:
            self.remove_interaction(name=r)

    ################################
    # Fix
    ################################

    def fix_variable(self):
        raise NotImplementedError

    ################################
    # PyQUBO
    ################################

    def from_pyqubo(self, expression):
        if not (isinstance(expression, pyqubo.Express) or isinstance(expression, pyqubo.Model)):
            raise TypeError(
                "'expression' must be a PyQUBO Expression (pyqubo.Express) or a PyQUBO Model (pyqubo.Model)."
            )
        raise NotImplementedError

    ################################
    # Constraints
    ################################

    def n_hot_constraint(self, target, n=1, strength=1.0, label=constants.DEFAULT_LABEL_N_HOT, delete_flag=False):
        self._check_argument_type("target", target, (pyqubo.Array, pyqubo.Spin, pyqubo.Binary))
        self._check_argument_type("n", n, int)
        self._check_argument_type("strength", strength, numbers.Number)
        self._check_argument_type("label", label, str)

        if not isinstance(target, pyqubo.Array):
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
                            self.add_interaction(
                                (avar, adj), name=f"{avar.label}*{adj.label} ({label})", coefficient=coeff
                            )
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
                            self.add_interaction(
                                (avar, adj), name=f"{avar.label}*{adj.label} ({label})", coefficient=coeff
                            )
                    for adj in original_variables:
                        coeff = -1.0 * strength
                        self.add_interaction((avar, adj), name=f"{avar.label}*{adj.label} ({label})", coefficient=coeff)

    def dependency_constraint(self, target_src, target_dst, strength=1.0, label=constants.DEFAULT_LABEL_DEPENDENCY):
        self._check_argument_type("strength", strength, numbers.Number)
        self._check_argument_type("label", label, str)
        raise NotImplementedError

    ################################
    # Utils
    ################################

    def merge(self, other):
        raise NotImplementedError

    def to_physical(self, placeholder={}):
        # TODO:
        # - resolve placeholder

        physical = PhysicalModel(mtype=self._mtype)

        linear, quadratic = {}, {}
        will_remove_linear, will_remove_quadratic = [], []

        # group by key
        for k, v in self._interactions[constants.INTERACTION_LINEAR].items():
            if not v["removed"]:
                # TODO: Calc difference from previous physical model by referencing dirty flags.
                if v["dirty"]:
                    self._interactions[constants.INTERACTION_LINEAR][k]["dirty"] = False
                if v["key"] in linear:
                    linear[v["key"]] += float(v["coefficient"] * v["scale"])
                else:
                    linear[v["key"]] = float(v["coefficient"] * v["scale"])
            else:
                will_remove_linear.append(k)
        for k, v in self._interactions[constants.INTERACTION_QUADRATIC].items():
            if not v["removed"]:
                # TODO: Calc difference from previous physical model by referencing dirty flags.
                if v["dirty"]:
                    self._interactions[constants.INTERACTION_QUADRATIC][k]["dirty"] = False
                if v["key"] in quadratic:
                    quadratic[v["key"]] += float(v["coefficient"] * v["scale"])
                else:
                    quadratic[v["key"]] = float(v["coefficient"] * v["scale"])
            else:
                will_remove_quadratic.append(k)

        # remove logically removed interactions
        for k in will_remove_linear:
            self._interactions[constants.INTERACTION_LINEAR].pop(k)
        for k in will_remove_quadratic:
            self._interactions[constants.INTERACTION_QUADRATIC].pop(k)

        # set to physical
        for k, v in linear.items():
            physical.add_interaction(k, body=constants.INTERACTION_LINEAR, coefficient=v)
        for k, v in quadratic.items():
            physical.add_interaction(k, body=constants.INTERACTION_QUADRATIC, coefficient=v)

        # save the last physical model
        self._previous_physical_model = physical

        return physical

    def _convert_mtype(self):
        """
        Converts the model to a QUBO model if the current model type is Ising, and vice versa.
        """
        raise NotImplementedError

    def to_ising(self):
        print(self._mtype)
        if self._mtype != constants.MODEL_ISING:
            self._convert_mtype()

    def to_qubo(self):
        if self._mtype != constants.MODEL_QUBO:
            self._convert_mtype()

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
        return self._deleted

    def get_fixed_array(self):
        """
        Returns a list of variables which are fixed.
        """
        return self._fixed

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
        s = "LogicalModel({"
        s += "'mtype': '" + str(self._mtype) + "', "
        s += "'variables': " + self.remove_leading_spaces(str(self._variables)) + ", "
        s += "'interactions': " + str(self._interactions) + ", "
        s += "'constraints': " + str(self._constraints) + "})"
        return s

    def __str__(self):
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
        s.append("┃  linear:")
        s.append(self.append_prefix(pprint.pformat(self._interactions[constants.INTERACTION_LINEAR]), length=4))
        s.append("┃  quadratic:")
        s.append(self.append_prefix(pprint.pformat(self._interactions[constants.INTERACTION_QUADRATIC]), length=4))
        s.append("┣━ constraints:")
        s.append(self.append_prefix(pprint.pformat(self._constraints), length=4))
        s.append("┗" + ("━" * 64))
        return "\n".join(s)
