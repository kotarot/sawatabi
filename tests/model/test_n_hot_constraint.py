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

import pytest

from sawatabi.model import NHotConstraint

################################
# N-Hot Constraint
################################


def test_n_hot_constraint():
    c = NHotConstraint()
    assert c.get_constraint_type() == "NHotConstraint"
    assert c.get_n() == 1
    assert c.get_strength() == 1.0
    assert c.get_label() == ""
    assert c.get_variables() == set()

    c.add_variable(set(["x0"]))
    assert c.get_variables() == set(["x0"])
    assert len(c.get_variables()) == 1

    c.add_variable(set(["x1", "x2"]))
    assert c.get_variables() == set(["x0", "x1", "x2"])
    assert len(c.get_variables()) == 3

    c.add_variable(set(["x0"]))
    assert c.get_variables() == set(["x0", "x1", "x2"])
    assert len(c.get_variables()) == 3

    c.add_variable(set(["x0", "x1"]))
    assert c.get_variables() == set(["x0", "x1", "x2"])
    assert len(c.get_variables()) == 3


def test_n_hot_constraint_constructors():
    c1 = NHotConstraint(n=2)
    assert c1.get_n() == 2
    assert c1.get_strength() == 1.0
    assert c1.get_label() == ""
    assert c1.get_variables() == set()

    c2 = NHotConstraint(strength=20)
    assert c2.get_n() == 1
    assert c2.get_strength() == 20.0
    assert c2.get_label() == ""
    assert c2.get_variables() == set()

    c3 = NHotConstraint(label="my label")
    assert c3.get_n() == 1
    assert c3.get_strength() == 1.0
    assert c3.get_label() == "my label"
    assert c3.get_variables() == set()

    c4 = NHotConstraint(variables=set(["a", "b", "c"]))
    assert c4.get_n() == 1
    assert c4.get_strength() == 1.0
    assert c4.get_label() == ""
    assert c4.get_variables() == set(["a", "b", "c"])


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


def test_n_hot_constraint_eq():
    assert NHotConstraint() == NHotConstraint()

    c1 = NHotConstraint(n=2, strength=20, label="my label", variables=set(["a", "b", "c"]))
    c2 = NHotConstraint(n=2, strength=20, label="my label", variables=set(["a", "b", "c"]))
    assert c1 == c2


def test_n_hot_constraint_ne():
    c = []
    c.append(NHotConstraint())
    c.append(NHotConstraint(n=2, strength=20, label="my label", variables=set(["a", "b", "c"])))
    c.append(NHotConstraint(n=2))
    c.append(NHotConstraint(strength=20))
    c.append(NHotConstraint(label="my label"))
    c.append(NHotConstraint(variables=set(["a", "b", "c"])))
    c.append("another type")

    for i in range(len(c) - 1):
        for j in range(i + 1, len(c)):
            assert c[i] != c[j]


def test_n_hot_constraint_repr():
    c = NHotConstraint()
    assert isinstance(c.__repr__(), str)
    assert "NHotConstraint({" in c.__repr__()
    assert "'constraint_type':" in c.__repr__()
    assert "'n':" in c.__repr__()
    assert "'strength':" in c.__repr__()
    assert "'label'" in c.__repr__()
    assert "'variables'" in c.__repr__()


def test_n_hot_constraint_str():
    c = NHotConstraint()
    assert isinstance(c.__str__(), str)
    assert "'constraint_type':" in c.__str__()
    assert "'n':" in c.__str__()
    assert "'strength':" in c.__str__()
    assert "'label'" in c.__str__()
    assert "'variables'" in c.__str__()
