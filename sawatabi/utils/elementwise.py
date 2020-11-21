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

import operator


def add(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(map(operator.add, a, b))
    elif isinstance(a, list) and isinstance(b, list):
        return list(map(operator.add, a, b))
    else:
        raise TypeError("a and b must be the same type and must be one of [tuple, list].")


def sub(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(map(operator.sub, a, b))
    elif isinstance(a, list) and isinstance(b, list):
        return list(map(operator.sub, a, b))
    else:
        raise TypeError("a and b must be the same type and must be one of [tuple, list].")