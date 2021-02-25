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

import numbers

import pyqubo

from sawatabi.base_mixin import BaseMixin
from sawatabi.utils.functions import Functions


class AbstractConstraint(BaseMixin):
    def __init__(self, label="", strength=1.0):
        self._constraint_class = None

        self._check_argument_type("label", label, str)
        if label == "":
            raise ValueError("'label' must not be empty.")
        self._check_argument_type("strength", strength, numbers.Number)

        self._label = label
        self._strength = strength

    def _check_variables_and_to_set(self, variables):
        self._check_argument_type("variables", variables, (pyqubo.Array, pyqubo.Spin, pyqubo.Binary, list, set))
        if isinstance(variables, (list, set)):
            self._check_argument_type_in_list("variables", variables, (pyqubo.Spin, pyqubo.Binary))
        if isinstance(variables, set):
            return variables
        if isinstance(variables, pyqubo.Array):
            variables = list(Functions._flatten(variables.bit_list))
        elif isinstance(variables, (pyqubo.Spin, pyqubo.Binary)):
            variables = [variables]
        return set(variables)

    def get_constraint_class(self):
        return self._constraint_class

    def get_label(self):
        return self._label

    def get_strength(self):
        return self._strength

    def to_model(self):
        raise NotImplementedError("#{self.class}##{__method__} must be implemented.")
