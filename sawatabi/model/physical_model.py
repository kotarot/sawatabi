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

import pprint

import sawatabi.constants as constants
from sawatabi.model.abstract_model import AbstractModel


class PhysicalModel(AbstractModel):
    def __init__(self, type=""):
        super().__init__(type)

    ################################
    # Interaction
    ################################

    def add_interaction(self, name, body, coefficient):
        self._interactions[body][name] = coefficient

    ################################
    # Built-in functions
    ################################

    def __repr__(self):
        s = "PhysicalModel({"
        s += "'type': '" + str(self._type) + "', "
        s += "'interactions': " + str(self._interactions) + "})"
        return s

    def __str__(self):
        s = []
        s.append("┏" + ("━" * 64))
        s.append("┃ PHYSICAL MODEL")
        s.append("┣" + ("━" * 64))
        s.append("┣━ type: " + str(self._type))
        s.append("┣━ interactions:")
        s.append("┃  linear:")
        s.append(self.append_prefix(pprint.pformat(self._interactions[constants.INTERACTION_BODY_LINEAR]), length=4))
        s.append("┃  quadratic:")
        s.append(self.append_prefix(pprint.pformat(self._interactions[constants.INTERACTION_BODY_QUADRATIC]), length=4))
        s.append("┗" + ("━" * 64))
        return "\n".join(s)
