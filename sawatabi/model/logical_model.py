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
            raise KeyError("Variables name '{}' is not defined in the model.".format(name))

        # tuple elementwise addition
        new_shape = tuple(map(sum, zip(self._variables[name].shape, shape)))
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
            raise ValueError(
                "An interaction named '{}' already exists. Cannot add the same name.".format(internal_name)
            )

        add_object = {
            "name": internal_name,
            "key": interaction_info["key"],
            "interacts": interaction_info["interacts"],
            "coefficient": coefficient,
            "scale": scale,
            "attributes": attributes,
            "timestamp": timestamp,
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
            raise KeyError(
                "An interaction named '{}' does not exist yet. Need to be added before updating.".format(internal_name)
            )

        # update if the value was given
        if coefficient is not None:
            self._interactions[body][internal_name]["coefficient"] = coefficient
        if scale is not None:
            self._interactions[body][internal_name]["scale"] = scale
        if attributes is not None:
            self._interactions[body][internal_name]["attributes"] = attributes
        self._interactions[body][internal_name]["timestamp"] = timestamp

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
                "An interaction named '{}' does not exist yet. Need to be added before updating.".format(internal_name)
            )

        # remove
        interaction = self._interactions[body].pop(internal_name)

        return interaction

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
                name = "{}*{}".format(target[0].label, target[1].label)
            else:
                interacts = (target[1], target[0])
                key = (target[1].label, target[0].label)
                name = "{}*{}".format(target[1].label, target[0].label)
        else:
            raise TypeError("Invalid 'target'.")
        return {"body": body, "interacts": interacts, "key": key, "name": name}

    ################################
    # Erase
    ################################

    def erase_variable(self):
        raise NotImplementedError

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

    def n_hot_constraint(self, target, n=1, scale=1.0, label=constants.DEFAULT_LABEL_N_HOT):
        self._check_argument_type("target", target, (pyqubo.Array, pyqubo.Spin, pyqubo.Binary))
        self._check_argument_type("n", n, int)
        self._check_argument_type("scale", scale, numbers.Number)
        self._check_argument_type("label", label, str)

        if not isinstance(target, pyqubo.Array):
            target = [target]
        for t in target:
            if label not in self._constraints:
                self._constraints[label] = NHotConstraint(n, scale, label)
            self._constraints[label].add(t.label)

    def dependency_constraint(self, target_src, target_dst, scale=1.0, label=constants.DEFAULT_LABEL_DEPENDENCY):
        self._check_argument_type("scale", scale, numbers.Number)
        self._check_argument_type("label", label, str)
        raise NotImplementedError

    ################################
    # Utils
    ################################

    def merge(self, other):
        raise NotImplementedError

    def convert_to_physical(self, placeholder={}):
        # TODO:
        # - resolve constraints
        # - resolve placeholder

        physical = PhysicalModel(mtype=self._mtype)

        linear, quadratic = {}, {}
        # group by key
        for k, v in self._interactions[constants.INTERACTION_LINEAR].items():
            if v["key"] in linear:
                linear[v["key"]] += float(v["coefficient"] * v["scale"])
            else:
                linear[v["key"]] = float(v["coefficient"] * v["scale"])
        for k, v in self._interactions[constants.INTERACTION_QUADRATIC].items():
            if v["key"] in quadratic:
                quadratic[v["key"]] += float(v["coefficient"] * v["scale"])
            else:
                quadratic[v["key"]] = float(v["coefficient"] * v["scale"])

        # set to physical
        for k, v in linear.items():
            physical.add_interaction(k, body=constants.INTERACTION_LINEAR, coefficient=v)
        for k, v in quadratic.items():
            physical.add_interaction(k, body=constants.INTERACTION_QUADRATIC, coefficient=v)

        return physical

    def convert_mtype(self):
        """
        Converts the model to a QUBO model if the current model type is Ising, and vice versa.
        """
        raise NotImplementedError

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

    def get_fixed_array(self):
        """
        Returns a list of variables which are fixed.
        """
        raise NotImplementedError

    def get_size(self):
        """
        Returns the number of all alive variables (i.e., variables which are not removed or fixed).
        """
        raise NotImplementedError

    def get_removed_size(self):
        """
        Returns the number of variables which are removed.
        """
        raise NotImplementedError

    def get_fixed_size(self):
        """
        Returns the number of variables which are fixed.
        """
        raise NotImplementedError

    def get_all_size(self):
        """
        Return the number of all variables including removed or fixed.
        """
        raise NotImplementedError

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
