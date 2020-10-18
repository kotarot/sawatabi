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
    def __init__(self, n=1, scale=1.0, label="", variables=[]):
        self._check_argument_type("n", n, int)
        self._check_argument_type("scale", scale, numbers.Number)
        self._check_argument_type("label", label, str)
        self._check_argument_type("variables", variables, list)
        super().__init__(scale, label, variables)
        self._n = n

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

    def add(self, variable):
        self._check_argument_type("variable", variable, (str, list))
        if isinstance(variable, str):
            self._variables.append(variable)
        elif isinstance(variable, list):
            self._variables.extend(variable)

    def get(self):
        return self._variables

    def __repr__(self):
        data = {
            "n": self._n,
            "scale": self._scale,
            "label": self._label,
            "variables": self._variables,
        }
        return str(data)
