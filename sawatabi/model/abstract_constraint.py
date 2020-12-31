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

from sawatabi.base_mixin import BaseMixin


class AbstractConstraint(BaseMixin):
    def __init__(self, strength=1.0, label="", variables=set()):
        self._constraint_type = None
        self._strength = strength
        self._label = label
        self._variables = variables

    def get_constraint_type(self):
        return self._constraint_type

    def get_strength(self):
        return self._strength

    def get_label(self):
        return self._label

    def get_variables(self):
        return self._variables
