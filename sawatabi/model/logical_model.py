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

import pyqubo

import sawatabi.constants as constants
from sawatabi.model.abstract_model import AbstractModel
from sawatabi.model.n_hot_constraint import NHotConstraint
from sawatabi.utils.functions import Functions
from sawatabi.utils.time import current_time_ms


class LogicalModel(AbstractModel):
    def __init__(self, type=""):
        super().__init__()
        if type in [constants.MODEL_ISING, constants.MODEL_QUBO]:
            self._type = type
        else:
            raise ValueError(
                "'type' must be one of {}.".format(
                    [constants.MODEL_ISING, constants.MODEL_QUBO]
                )
            )

    ################################
    # Private static methods
    ################################

    @staticmethod
    def _check_argument_type(name, value, type):
        if not isinstance(value, type):
            if isinstance(type, tuple):
                typestr = [t.__name__ for t in type]
                article = "one of"
            else:
                typestr = type.__name__
                if typestr[0] in ["a", "e", "i", "o", "u"]:
                    article = "an"
                else:
                    article = "a"
            raise TypeError("'{}' must be {} {}.".format(name, article, typestr))

    @staticmethod
    def _check_argument_for_shape(shape):
        if len(shape) == 0:
            raise TypeError("'shape' must not be an empty tuple.")
        for i in shape:
            if not isinstance(i, int):
                raise TypeError("All elements of 'shape' must be an integer.")

    @staticmethod
    def _modeltype_to_vartype(modeltype):
        if modeltype == constants.MODEL_ISING:
            vartype = "SPIN"
        elif modeltype == constants.MODEL_QUBO:
            vartype = "BINARY"
        else:
            raise ValueError("Invalid 'modeltype'")
        return vartype

    @staticmethod
    def _vartype_to_modeltype(vartype):
        if vartype == "SPIN":
            modeltype = constants.MODEL_ISING
        elif vartype == "BINARY":
            modeltype = constants.MODEL_QUBO
        else:
            raise ValueError("Invalid 'vartype'")
        return modeltype

    @staticmethod
    def _get_interaction_body_from_target(target):
        if isinstance(target, (pyqubo.Spin, pyqubo.Binary)):
            body = constants.INTERACTION_1_BODY
        elif isinstance(target, tuple):
            if len(target) != 2:
                raise TypeError("The length of a tuple 'target' must be two.")
            for i in target:
                if not isinstance(i, (pyqubo.Spin, pyqubo.Binary)):
                    raise TypeError(
                        "All elements of 'target' must be a 'pyqubo.Spin' or 'pyqubo.Binary'."
                    )
            body = constants.INTERACTION_2_BODY
        else:
            raise TypeError("Invalid 'target'.")
        return body

    @staticmethod
    def _get_default_name_of_interaction(interaction_body, target):
        if interaction_body == constants.INTERACTION_1_BODY:
            name = target.label
        elif interaction_body == constants.INTERACTION_2_BODY:
            # To dictionary order
            if target[0].label < target[1].label:
                this_target = (target[0].label, target[1].label)
            else:
                this_target = (target[1].label, target[0].label)
            name = str(this_target)
        return name

    ################################
    # Variables
    ################################

    def variables(self, name, shape=()):
        if isinstance(name, pyqubo.Array):
            flattened = list(Functions._flatten(name.bit_list))
            if (
                (self._type == constants.MODEL_ISING)
                and isinstance(flattened[0], pyqubo.Binary)
            ) or (
                (self._type == constants.MODEL_QUBO)
                and isinstance(flattened[0], pyqubo.Spin)
            ):
                raise TypeError("Model type and PyQUBO Array type mismatch.")

            # Retrieve label from the pyqubo variable
            found = flattened[0].label.index("[")
            this_name = flattened[0].label[:found]

            self._variables[this_name] = name
            return self._variables[this_name]

        self._check_argument_type("name", name, str)
        self._check_argument_type("shape", shape, tuple)
        self._check_argument_for_shape(shape)

        vartype = self._modeltype_to_vartype(self._type)

        self._variables[name] = pyqubo.Array.create(name, shape=shape, vartype=vartype)
        return self._variables[name]

    def append(self, name, shape=()):
        self._check_argument_type("name", name, str)
        self._check_argument_type("shape", shape, tuple)
        self._check_argument_for_shape(shape)

        if name not in self._variables:
            raise KeyError(
                "Variables name '{}' is not defined in the model.".format(name)
            )

        # tuple elementwise addition
        new_shape = tuple(map(sum, zip(self._variables[name].shape, shape)))
        vartype = self._modeltype_to_vartype(self._type)

        self._variables[name] = pyqubo.Array.create(
            name, shape=new_shape, vartype=vartype
        )
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

        if name:
            # Already given the specific name
            self._check_argument_type("name", name, str)
            new_name = name
        else:
            # Will be automatically named by the default name
            body = self._get_interaction_body_from_target(target)
            new_name = self._get_default_name_of_interaction(body, target)

        add_object = {
            "name": new_name,
            "coefficient": coefficient,
            "scale": scale,
            "attributes": attributes,
            "timestamp": timestamp,
        }
        self._interactions[new_name] = add_object
        return add_object

    ################################
    # Update
    ################################

    def update_interaction(
        self,
        target=None,
        name="",
        coefficient=0.0,
        scale=1.0,
        attributes={},
        timestamp=current_time_ms(),
    ):
        if (not target) and (not name):
            raise ValueError("Either 'target' or 'name' must be specified.")
        if target and name:
            raise ValueError(
                "Both 'target' and 'name' cannot be specified simultaneously."
            )

        self._check_argument_type("coefficient", coefficient, numbers.Number)
        self._check_argument_type("scale", scale, numbers.Number)
        self._check_argument_type("attributes", attributes, dict)
        self._check_argument_type("timestamp", timestamp, int)

        if name:
            # Already given the specific name
            self._check_argument_type("name", name, str)
            new_name = name
        else:
            # Will be automatically named by the default name
            body = self._get_interaction_body_from_target(target)
            new_name = self._get_default_name_of_interaction(body, target)

        if new_name not in self._interactions:
            raise KeyError(
                "An interaction named '{}' does not exist yet. Need to be added before updating.".format(
                    new_name
                )
            )

        # TODO: Need to change only updated values.
        update_object = {
            "name": new_name,
            "coefficient": coefficient,
            "scale": scale,
            "attributes": attributes,
            "timestamp": timestamp,
        }
        self._interactions[new_name] = update_object
        return update_object

    ################################
    # Remove
    ################################

    def remove_interaction(self):
        raise NotImplementedError

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
        if not (
            isinstance(expression, pyqubo.Express)
            or isinstance(expression, pyqubo.Model)
        ):
            raise TypeError(
                "'expression' must be a PyQUBO Expression (pyqubo.Express) or a PyQUBO Model (pyqubo.Model)."
            )
        raise NotImplementedError

    ################################
    # Constraints
    ################################

    def n_hot_constraint(
        self, target, n=1, scale=1.0, label=constants.DEFAULT_LABEL_N_HOT
    ):
        self._check_argument_type(
            "target", target, (pyqubo.Array, pyqubo.Spin, pyqubo.Binary)
        )
        self._check_argument_type("n", n, int)
        self._check_argument_type("scale", scale, numbers.Number)
        self._check_argument_type("label", label, str)

        if not isinstance(target, pyqubo.Array):
            target = [target]
        for t in target:
            if label not in self._constraints:
                self._constraints[label] = NHotConstraint(n, scale, label)
            self._constraints[label].add(t.label)

    def dependency_constraint(
        self,
        target_src,
        target_dst,
        scale=1.0,
        label=constants.DEFAULT_LABEL_DEPENDENCY,
    ):
        self._check_argument_type("scale", scale, numbers.Number)
        self._check_argument_type("label", label, str)
        raise NotImplementedError

    ################################
    # Utils
    ################################

    def merge(self, other):
        raise NotImplementedError

    def convert_to_physical(self, placeholder={}):
        raise NotImplementedError

    def convert_type(self):
        """
        Converts the model to a QUBO model if the current model type is Ising, and vice versa.
        """
        raise NotImplementedError

    ################################
    # Getters
    ################################

    def get_type(self):
        return self._type

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
        s = []
        s.append("┏━━━━━━━━━━━━━━━━━━━━━━━━")
        s.append("┃ LOGICAL MODEL")
        s.append("┣━━━━━━━━━━━━━━━━━━━━━━━━")
        s.append("┣━ type: " + str(self._type))
        s.append("┣┳ variables: " + str(list(self._variables.keys())))
        for name, vars in self._variables.items():
            s.append(" ┗ name: " + name)
            s.append(str(vars))
        s.append("┣━ interactions: " + str(self._interactions))
        s.append("┗━ constraints: " + str(self._constraints))
        return "\n".join(s)
