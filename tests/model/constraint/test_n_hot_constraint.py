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

import pyqubo
import pytest

from sawatabi.model.constraint import NHotConstraint

################################
# N-Hot Constraint
################################


def test_n_hot_constraint():
    c = NHotConstraint()
    assert c.get_constraint_class() == "NHotConstraint"
    assert c.get_variables() == set()
    assert c.get_n() == 1
    assert c.get_label() == "Default N-hot Constraint"
    assert c.get_strength() == 1.0

    x0 = pyqubo.Spin("x0")
    x1 = pyqubo.Spin("x1")
    x2 = pyqubo.Binary("x2")

    c.add_variable(x0)
    assert c.get_variables() == set([x0])
    assert len(c.get_variables()) == 1

    c.add_variable([x1, x2])
    assert c.get_variables() == set([x0, x1, x2])
    assert len(c.get_variables()) == 3

    c.add_variable(set([x0]))
    assert c.get_variables() == set([x0, x1, x2])
    assert len(c.get_variables()) == 3

    c.add_variable(variables=[x0, x1])
    assert c.get_variables() == set([x0, x1, x2])
    assert len(c.get_variables()) == 3

    c.remove_variable(variables=[x0])
    assert c.get_variables() == set([x1, x2])
    assert len(c.get_variables()) == 2

    with pytest.raises(ValueError):
        c.remove_variable(variables=[x0])


def test_n_hot_constraint_constructors():
    a = pyqubo.Array.create("a", shape=(2, 2), vartype="SPIN")

    c1 = NHotConstraint(variables=a)
    assert c1.get_variables() == set([a[0, 0], a[0, 1], a[1, 0], a[1, 1]])
    assert c1.get_n() == 1
    assert c1.get_label() == "Default N-hot Constraint"
    assert c1.get_strength() == 1.0

    c2 = NHotConstraint(n=2)
    assert c2.get_variables() == set()
    assert c2.get_n() == 2
    assert c2.get_label() == "Default N-hot Constraint"
    assert c2.get_strength() == 1.0

    c3 = NHotConstraint(label="my label")
    assert c3.get_variables() == set()
    assert c3.get_n() == 1
    assert c3.get_label() == "my label"
    assert c3.get_strength() == 1.0

    c4 = NHotConstraint(strength=20)
    assert c4.get_variables() == set()
    assert c4.get_n() == 1
    assert c4.get_label() == "Default N-hot Constraint"
    assert c4.get_strength() == 20.0


@pytest.mark.parametrize("n", [-10, 0])
def test_n_hot_constraint_valueerror(n):
    with pytest.raises(ValueError):
        NHotConstraint(n=n)

    with pytest.raises(ValueError):
        NHotConstraint(label="")


def test_n_hot_constraint_typeerror():
    with pytest.raises(TypeError):
        NHotConstraint(variables="invalid type")

    with pytest.raises(TypeError):
        NHotConstraint(variables=set([1, 2, 3]))

    with pytest.raises(TypeError):
        NHotConstraint(n="invalid type")

    with pytest.raises(TypeError):
        NHotConstraint(n=1.0)

    with pytest.raises(TypeError):
        NHotConstraint(label=12345)

    with pytest.raises(TypeError):
        NHotConstraint(strength="invalid type")


################################
# Built-in functions
################################


def test_n_hot_constraint_eq():
    assert NHotConstraint() == NHotConstraint()

    a = pyqubo.Spin("a")
    b = pyqubo.Binary("b")
    c1 = NHotConstraint(variables=set([a, b]), n=2, label="my label", strength=20)
    c2 = NHotConstraint(variables=[a, b], n=2, label="my label", strength=2 * 10)
    assert c1 == c2


def test_n_hot_constraint_ne():
    a = pyqubo.Spin("a")
    b = pyqubo.Binary("b")
    c = []
    c.append(NHotConstraint())
    c.append(NHotConstraint(variables=[a, b], n=2, label="my label", strength=20))
    c.append(NHotConstraint(variables=set([a, b])))
    c.append(NHotConstraint(n=2))
    c.append(NHotConstraint(label="my label"))
    c.append(NHotConstraint(strength=20))
    c.append("another type")

    for i in range(len(c) - 1):
        for j in range(i + 1, len(c)):
            assert c[i] != c[j]


def test_n_hot_constraint_repr():
    c = NHotConstraint()
    assert isinstance(c.__repr__(), str)
    assert "NHotConstraint({" in c.__repr__()
    assert "'constraint_class':" in c.__repr__()
    assert "'variables'" in c.__repr__()
    assert "'n':" in c.__repr__()
    assert "'label'" in c.__repr__()
    assert "'strength':" in c.__repr__()


def test_n_hot_constraint_str():
    c = NHotConstraint()
    assert isinstance(c.__str__(), str)
    assert "'constraint_class':" in c.__str__()
    assert "'variables'" in c.__str__()
    assert "'n':" in c.__str__()
    assert "'label'" in c.__str__()
    assert "'strength':" in c.__str__()
