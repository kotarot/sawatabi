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

from ..constants import *
from .abstract_model import *

class LogicalModel(AbstractModel):
    def __init__(self, type=''):
        super().__init__()
        if type in [MODEL_TYPE_ISING, MODEL_TYPE_QUBO]:
            self.type = type
        else:
            raise ValueError("type must be one of {}.".format([MODEL_TYPE_ISING, MODEL_TYPE_QUBO]))

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

    def select_variable(self):
        raise NotImplementedError

    ################################
    # Update
    ################################

    def update_variable(self):
        raise NotImplementedError

    def update_variable(self):
        raise NotImplementedError

    ################################
    # Remove
    ################################

    def remove_variable(self):
        raise NotImplementedError

    def remove_variable(self):
        raise NotImplementedError

    ################################
    # Fix
    ################################

    def fix_variable(self):
        raise NotImplementedError

    def fix_variable(self):
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

    ################################
    # Others
    ################################
    def get_array(self):
        """
        Returns a list of alive variables (i.e., variables which are not removed nor fixed).
        """
        raise NotImplementedError

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
        s.append('--- Logical Model ---')
        s.append('  type: ' + self.type)
        s.append('  variables: ' + str(self.variables))
        s.append('  interactions: ' + str(self.interactions))
        return '\n'.join(s)
