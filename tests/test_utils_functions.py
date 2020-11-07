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

import pytest

from sawatabi.utils.functions import Functions


@pytest.mark.parametrize(
    "a,b,y", [((2,), (1,), (3,)), ((44, 33), (22, 11), (66, 44)), ((66, 55, 44), (33, 22, 11), (99, 77, 55))]
)
def test_elementwise_add_tuple(a, b, y):
    x = Functions.elementwise_add(a, b)
    assert type(x) is tuple
    assert x == y


@pytest.mark.parametrize(
    "a,b,y", [((2,), (1,), (1,)), ((44, 33), (22, 11), (22, 22)), ((66, 55, 44), (33, 22, 11), (33, 33, 33))]
)
def test_elementwise_sub_tuple(a, b, y):
    x = Functions.elementwise_sub(a, b)
    assert type(x) is tuple
    assert x == y


@pytest.mark.parametrize(
    "a,b,y", [([2], [1], [3]), ([44, 33], [22, 11], [66, 44]), ([66, 55, 44], [33, 22, 11], [99, 77, 55])]
)
def test_elementwise_add_list(a, b, y):
    x = Functions.elementwise_add(a, b)
    assert type(x) is list
    assert x == y


@pytest.mark.parametrize(
    "a,b,y", [([2], [1], [1]), ([44, 33], [22, 11], [22, 22]), ([66, 55, 44], [33, 22, 11], [33, 33, 33])]
)
def test_elementwise_sub_list(a, b, y):
    x = Functions.elementwise_sub(a, b)
    assert type(x) is list
    assert x == y


def test_elementwise_invalid_type():
    with pytest.raises(TypeError):
        Functions.elementwise_add("a", "b")

    with pytest.raises(TypeError):
        Functions.elementwise_sub("a", "b")
