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

import sawatabi
import sawatabi.constants as constants
from sawatabi.model.constraint.abstract_constraint import AbstractConstraint


class EqualityConstraint(AbstractConstraint):
    def __init__(self, variables_1=None, variables_2=None, label=constants.DEFAULT_LABEL_EQUALITY, strength=1.0):
        super().__init__(label=label, strength=strength)
        self._constraint_class = self.__class__.__name__

        # Avoid duplicate variable, so we use set() for variables
        if variables_1 is None:
            self._variables_1 = set()
        else:
            self._variables_1 = self._check_variables_and_to_set(variables_1)
        if variables_2 is None:
            self._variables_2 = set()
        else:
            self._variables_2 = self._check_variables_and_to_set(variables_2)

    def add_variable_to_1(self, variables):
        variables_set = self._check_variables_and_to_set(variables)
        self._variables_1 = self._variables_1.union(variables_set)

    def add_variable_to_2(self, variables):
        variables_set = self._check_variables_and_to_set(variables)
        self._variables_2 = self._variables_2.union(variables_set)

    def remove_variable_from_1(self, variables):
        variables_set = self._check_variables_and_to_set(variables)
        for v in variables_set:
            if v not in self._variables_1:
                raise ValueError(f"Variable '{v}' does not exist in the constraint variables.")
        self._variables_1 = self._variables_1.difference(variables_set)

    def remove_variable_from_2(self, variables):
        variables_set = self._check_variables_and_to_set(variables)
        for v in variables_set:
            if v not in self._variables_2:
                raise ValueError(f"Variable '{v}' does not exist in the constraint variables.")
        self._variables_2 = self._variables_2.difference(variables_set)

    def get_variables_1(self):
        return self._variables_1

    def get_variables_2(self):
        return self._variables_2

    def to_model(self):
        model = sawatabi.model.LogicalModel(mtype="qubo")

        # Equality constraint:
        #   E = ( \sum{ x_i } - \sum{ y_i } )^2
        for var in self._variables_1.union(self._variables_2):
            coeff = -1.0 * self._strength
            model.add_interaction(var, name=f"{var.label} ({self._label})", coefficient=coeff)
        for var in self._variables_1:
            for adj in self._variables_1:
                if var.label < adj.label:
                    coeff = -2.0 * self._strength
                    model.add_interaction((var, adj), name=f"{var.label}*{adj.label} ({self._label})", coefficient=coeff)
        for var in self._variables_2:
            for adj in self._variables_2:
                if var.label < adj.label:
                    coeff = -2.0 * self._strength
                    model.add_interaction((var, adj), name=f"{var.label}*{adj.label} ({self._label})", coefficient=coeff)
        for var in self._variables_1:
            for adj in self._variables_2:
                coeff = 2.0 * self._strength
                model.add_interaction((var, adj), name=f"{var.label}*{adj.label} ({self._label})", coefficient=coeff)

        return model

    ################################
    # Built-in functions
    ################################

    def __eq__(self, other):
        return (
            isinstance(other, EqualityConstraint)
            and (self._constraint_class == other._constraint_class)
            and (self._variables_1 == other._variables_1)
            and (self._variables_2 == other._variables_2)
            and (self._label == other._label)
            and (self._strength == other._strength)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__str__()})"

    def __str__(self):
        data = {
            "constraint_class": self._constraint_class,
            "variables_1": self._variables_1,
            "variables_2": self._variables_2,
            "label": self._label,
            "strength": self._strength,
        }
        return str(data)
