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

from sawatabi.model import NHotConstraint


################################
# N-Hot Constraint
################################


def test_n_hot_constraint():
    c = NHotConstraint()
    assert c._n == 1
    assert c._scale == 1.0
    assert c._label == ""
    assert c._variables == []

    c.add("x0")
    assert c._variables == ["x0"]
    assert len(c._variables) == 1

    c.add(["x1", "x2"])
    assert c._variables == ["x0", "x1", "x2"]
    assert len(c._variables) == 3


def test_n_hot_constraint_constructors():
    c1 = NHotConstraint(n=2)
    assert c1._n == 2
    assert c1._scale == 1.0
    assert c1._label == ""
    assert c1._variables == []

    c2 = NHotConstraint(scale=20)
    assert c2._n == 1
    assert c2._scale == 20.0
    assert c2._label == ""
    assert c2._variables == []

    c3 = NHotConstraint(label="my label")
    assert c3._n == 1
    assert c3._scale == 1.0
    assert c3._label == "my label"
    assert c3._variables == []

    c4 = NHotConstraint(variables=["a", "b", "c"])
    assert c4._n == 1
    assert c4._scale == 1.0
    assert c4._label == ""
    assert c4._variables == ["a", "b", "c"]


def test_n_hot_constraint_constructors():
    with pytest.raises(TypeError):
        NHotConstraint(n="invalid type")

    with pytest.raises(TypeError):
        NHotConstraint(n=1.0)

    with pytest.raises(TypeError):
        NHotConstraint(scale="invalid type")

    with pytest.raises(TypeError):
        NHotConstraint(label=12345)

    with pytest.raises(TypeError):
        NHotConstraint(variables="invalid type")

    #with pytest.raises(TypeError):
    #    NHotConstraint(variables=[1, 2, 3])
