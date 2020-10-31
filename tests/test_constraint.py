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
    assert c._constraint_type == "NHotConstraint"
    assert c._n == 1
    assert c._strength == 1.0
    assert c._label == ""
    assert c._variables == set()

    c.add(set(["x0"]))
    assert c._variables == set(["x0"])
    assert len(c._variables) == 1

    c.add(set(["x1", "x2"]))
    assert c._variables == set(["x0", "x1", "x2"])
    assert len(c._variables) == 3

    c.add(set(["x0"]))
    assert c._variables == set(["x0", "x1", "x2"])
    assert len(c._variables) == 3

    c.add(set(["x0", "x1"]))
    assert c._variables == set(["x0", "x1", "x2"])
    assert len(c._variables) == 3

    assert c.get() == set(["x0", "x1", "x2"])
    assert len(c.get()) == 3


def test_n_hot_constraint_constructors():
    c1 = NHotConstraint(n=2)
    assert c1._n == 2
    assert c1._strength == 1.0
    assert c1._label == ""
    assert c1._variables == set()

    c2 = NHotConstraint(strength=20)
    assert c2._n == 1
    assert c2._strength == 20.0
    assert c2._label == ""
    assert c2._variables == set()

    c3 = NHotConstraint(label="my label")
    assert c3._n == 1
    assert c3._strength == 1.0
    assert c3._label == "my label"
    assert c3._variables == set()

    c4 = NHotConstraint(variables=set(["a", "b", "c"]))
    assert c4._n == 1
    assert c4._strength == 1.0
    assert c4._label == ""
    assert c4._variables == set(["a", "b", "c"])


@pytest.mark.parametrize("n", [-10, 0])
def test_n_hot_constraint_valueerror(n):
    with pytest.raises(ValueError):
        NHotConstraint(n=n)


def test_n_hot_constraint_typeerror():
    with pytest.raises(TypeError):
        NHotConstraint(n="invalid type")

    with pytest.raises(TypeError):
        NHotConstraint(n=1.0)

    with pytest.raises(TypeError):
        NHotConstraint(strength="invalid type")

    with pytest.raises(TypeError):
        NHotConstraint(label=12345)

    with pytest.raises(TypeError):
        NHotConstraint(variables="invalid type")

    # TODO: This error should be raises, but not implemented yet.
    # with pytest.raises(TypeError):
    #     NHotConstraint(variables=set([1, 2, 3]))


################################
# Built-in functions
################################


def test_logical_model_repr():
    c = NHotConstraint()
    assert isinstance(c.__repr__(), str)
    assert "NHotConstraint({" in c.__repr__()
    assert "'constraint_type':" in c.__repr__()
    assert "'n':" in c.__repr__()
    assert "'strength':" in c.__repr__()
    assert "'label'" in c.__repr__()
    assert "'variables'" in c.__repr__()


def test_logical_model_str():
    c = NHotConstraint()
    assert isinstance(c.__str__(), str)
    assert "'constraint_type':" in c.__str__()
    assert "'n':" in c.__str__()
    assert "'strength':" in c.__str__()
    assert "'label'" in c.__str__()
    assert "'variables'" in c.__str__()
