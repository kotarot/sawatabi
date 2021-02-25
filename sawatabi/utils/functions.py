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

import operator

import numpy as np


class Functions:
    @classmethod
    def _flatten(cls, lst):
        for element in lst:
            if isinstance(element, list):
                yield from cls._flatten(element)
            else:
                yield element

    @classmethod
    def elementwise_add(cls, a, b):
        if isinstance(a, tuple) and isinstance(b, tuple):
            return tuple(map(operator.add, a, b))
        if isinstance(a, list) and isinstance(b, list):
            return list(map(operator.add, a, b))
        raise TypeError("a and b must be the same type and must be one of [tuple, list].")

    @classmethod
    def elementwise_sub(cls, a, b):
        if isinstance(a, tuple) and isinstance(b, tuple):
            return tuple(map(operator.sub, a, b))
        if isinstance(a, list) and isinstance(b, list):
            return list(map(operator.sub, a, b))
        raise TypeError("a and b must be the same type and must be one of [tuple, list].")

    @classmethod
    def elementwise_max(cls, a, b):
        if isinstance(a, tuple) and isinstance(b, tuple):
            return tuple(np.maximum(a, b))
        if isinstance(a, list) and isinstance(b, list):
            return list(np.maximum(a, b))
        raise TypeError("a and b must be the same type and must be one of [tuple, list].")

    @classmethod
    def elementwise_min(cls, a, b):
        if isinstance(a, tuple) and isinstance(b, tuple):
            return tuple(np.minimum(a, b))
        if isinstance(a, list) and isinstance(b, list):
            return list(np.minimum(a, b))
        raise TypeError("a and b must be the same type and must be one of [tuple, list].")
