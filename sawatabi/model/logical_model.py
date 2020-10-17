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

from sawatabi.constants import MODEL_TYPE_ISING, MODEL_TYPE_QUBO
from sawatabi.model.abstract_model import AbstractModel
from sawatabi.utils.time import current_time_ms


class LogicalModel(AbstractModel):
    def __init__(self, type=""):
        super().__init__()
        if type in [MODEL_TYPE_ISING, MODEL_TYPE_QUBO]:
            self._type = type
        else:
            raise ValueError(
                "'type' must be one of {}.".format([MODEL_TYPE_ISING, MODEL_TYPE_QUBO])
            )

    ################################
    # Array
    ################################

    def array(self, name, shape=()):
        if isinstance(name, pyqubo.Array):
            self._array = name
            self._array_name = "TODO"
            return self._array
        self._check_argument_for_name(name)
        self._check_argument_for_shape(shape)

        vartype = self._modeltype_to_vartype(self._type)

        self._array = pyqubo.Array.create(name, shape=shape, vartype=vartype)
        self._array_name = name
        return self._array

    def append(self, shape=()):
        self._check_argument_for_shape(shape)

        new_shape = tuple(
            map(sum, zip(self._array.shape, shape))
        )  # tuple elementwise addition
        vartype = self._modeltype_to_vartype(self._type)

        self._array = pyqubo.Array.create(
            self._array_name, shape=new_shape, vartype=vartype
        )
        return self._array

    @staticmethod
    def _check_argument_for_name(name):
        if not isinstance(name, str):
            raise TypeError("'name' must be a string.")

    @staticmethod
    def _check_argument_for_shape(shape):
        if not isinstance(shape, tuple):
            raise TypeError("'shape' must be a tuple.")
        else:
            if len(shape) == 0:
                raise TypeError("'shape' must not be an empty tuple.")
            for i in shape:
                if not isinstance(i, int):
                    raise TypeError("All elements of 'shape' must be an integer.")

    @staticmethod
    def _modeltype_to_vartype(modeltype):
        if modeltype == MODEL_TYPE_ISING:
            vartype = "SPIN"
        elif modeltype == MODEL_TYPE_QUBO:
            vartype = "BINARY"
        else:
            raise ValueError("Invalid 'modeltype'")
        return vartype

    @staticmethod
    def _vartype_to_modeltype(vartype):
        if vartype == "SPIN":
            modeltype = MODEL_TYPE_ISING
        elif vartype == "BINARY":
            modeltype = MODEL_TYPE_QUBO
        else:
            raise ValueError("Invalid 'vartype'")
        return modeltype

    ################################
    # Add
    ################################

    def add_variable(self):
        raise NotImplementedError

    def add_variables(self):
        raise NotImplementedError

    def add_interaction(self):
        raise NotImplementedError

    def add_interactions(self):
        raise NotImplementedError

    ################################
    # Select
    ################################

    def select_variable(self):
        raise NotImplementedError

    def select_interaction(self):
        raise NotImplementedError

    ################################
    # Update
    ################################

    def update_variable(
        self,
        name,
        coefficient=0.0,
        scale=1.0,
        attributes={},
        timestamp=current_time_ms(),
    ):
        this_name = name.label
        self._variables[this_name] = {
            "name": this_name,
            "coefficient": coefficient,
            "scale": scale,
            "attributes": attributes,
            "timestamp": timestamp,
        }

    def update_interaction(
        self,
        name,
        coefficient=0.0,
        scale=1.0,
        attributes={},
        timestamp=current_time_ms(),
    ):
        # To dictionary order
        if name[0].label < name[1].label:
            this_name = (name[0].label, name[1].label)
        else:
            this_name = (name[1].label, name[0].label)
        self._interactions[this_name] = {
            "name": this_name,
            "coefficient": coefficient,
            "scale": scale,
            "attributes": attributes,
            "timestamp": timestamp,
        }

    ################################
    # Remove
    ################################

    def remove_variable(self):
        raise NotImplementedError

    def remove_interaction(self):
        raise NotImplementedError

    ################################
    # Fix
    ################################

    def fix_variable(self):
        raise NotImplementedError

    def fix_interaction(self):
        raise NotImplementedError

    ################################
    # PyQUBO
    ################################

    def from_pyqubo(self, expression):
        if not isinstance(expression, pyqubo.Express):
            raise TypeError(
                "'expression' must be a PyQUBO Expression (pyqubo.Express)."
            )
        raise NotImplementedError

    ################################
    # Constraints
    ################################

    def n_hot_constraint(self):
        raise NotImplementedError

    def dependency_constraint(self):
        raise NotImplementedError

    ################################
    # Utils
    ################################

    def merge(self):
        raise NotImplementedError

    def convert_to_physical(self):
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

    def get_array(self):
        """
        Returns a list of alive variables (i.e., variables which are not removed nor fixed).
        """
        return self._array

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

    ################################
    # Built-in functions
    ################################

    def __repr__(self):
        s = []
        s.append("[Logical Model]")
        s.append("type: " + self._type)
        s.append("array: " + str(self._array.shape))
        s.append(str(self._array))
        s.append("variables: " + str(self._variables))
        s.append("interactions: " + str(self._interactions))
        return "\n".join(s)
