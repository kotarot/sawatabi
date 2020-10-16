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
            return self._array

        if not isinstance(name, str):
            raise TypeError("'name' must be a string.")
        if not isinstance(shape, tuple):
            raise TypeError("'shape' must be a tuple.")
        else:
            if len(shape) == 0:
                raise TypeError("'shape' must not be an empty tuple.")
            for i in shape:
                if not isinstance(i, int):
                    raise TypeError("All elements of 'shape' must be an integer.")

        if self._type == MODEL_TYPE_ISING:
            vartype = "SPIN"
        elif self._type == MODEL_TYPE_QUBO:
            vartype = "BINARY"

        self._array = pyqubo.Array.create(name, shape=shape, vartype=vartype)
        return self._array

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

    def update_variable(self):
        raise NotImplementedError

    def update_interaction(self):
        raise NotImplementedError

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

    def add_from_pyqubo(self):
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
        s.append("--- Logical Model ---")
        s.append("type: " + self._type)
        s.append("array: " + str(self._array.shape))
        s.append(str(self._array))
        s.append("variables: " + str(self._variables))
        s.append("interactions: " + str(self._interactions))
        return "\n".join(s)
