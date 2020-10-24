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

from sawatabi.model.abstract_constraint import AbstractConstraint


class NHotConstraint(AbstractConstraint):
    def __init__(self, n=1, strength=1.0, label="", variables=None):
        self._check_argument_type("n", n, int)
        if n <= 0:
            raise ValueError("'n' must be a positive integer.")
        self._check_argument_type("strength", strength, numbers.Number)
        self._check_argument_type("label", label, str)
        if variables is None:
            # to avoid to share the identical set between different Constraint classes...
            variables = set()
        self._check_argument_type("variables", variables, set)
        super().__init__(strength, label, variables)
        self._n = n

    def add(self, variable):
        self._check_argument_type("variable", variable, set)
        # avoid duplicate variable (so we use set())
        self._variables = self._variables.union(variable)

    def get(self):
        return self._variables

    def __repr__(self):
        return f"NHotConstraint({self.__str__()})"

    def __str__(self):
        data = {
            "n": self._n,
            "strength": self._strength,
            "label": self._label,
            "variables": self._variables,
        }
        return str(data)
