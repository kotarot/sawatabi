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

from sawatabi.model.constraint import ZeroOrOneHotConstraint

################################
# Zero-or-One-Hot Constraint
################################


def test_zero_or_one_hot_constraint():
    c = ZeroOrOneHotConstraint()
    assert c.get_constraint_class() == "ZeroOrOneHotConstraint"
    assert c.get_variables() == set()
    assert c.get_label() == "Default Zero-or-One-hot Constraint"
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


def test_zero_or_one_hot_constraint_constructors():
    a = pyqubo.Array.create("a", shape=(2, 2), vartype="SPIN")

    c1 = ZeroOrOneHotConstraint(variables=a)
    assert c1.get_variables() == set([a[0, 0], a[0, 1], a[1, 0], a[1, 1]])
    assert c1.get_label() == "Default Zero-or-One-hot Constraint"
    assert c1.get_strength() == 1.0

    c2 = ZeroOrOneHotConstraint(label="my label")
    assert c2.get_variables() == set()
    assert c2.get_label() == "my label"
    assert c2.get_strength() == 1.0

    c3 = ZeroOrOneHotConstraint(strength=20)
    assert c3.get_variables() == set()
    assert c3.get_label() == "Default Zero-or-One-hot Constraint"
    assert c3.get_strength() == 20.0


def test_zero_or_one_hot_constraint_typeerror():
    with pytest.raises(TypeError):
        ZeroOrOneHotConstraint(variables="invalid type")

    with pytest.raises(TypeError):
        ZeroOrOneHotConstraint(variables=set([1, 2, 3]))

    with pytest.raises(TypeError):
        ZeroOrOneHotConstraint(label=12345)

    with pytest.raises(TypeError):
        ZeroOrOneHotConstraint(strength="invalid type")


################################
# Built-in functions
################################


def test_zero_or_one_hot_constraint_eq():
    assert ZeroOrOneHotConstraint() == ZeroOrOneHotConstraint()

    a = pyqubo.Spin("a")
    b = pyqubo.Binary("b")
    c1 = ZeroOrOneHotConstraint(variables=set([a, b]), label="my label", strength=20)
    c2 = ZeroOrOneHotConstraint(variables=[a, b], label="my label", strength=2 * 10)
    assert c1 == c2


def test_zero_or_one_hot_constraint_ne():
    a = pyqubo.Spin("a")
    b = pyqubo.Binary("b")
    c = []
    c.append(ZeroOrOneHotConstraint())
    c.append(ZeroOrOneHotConstraint(variables=[a, b], label="my label", strength=20))
    c.append(ZeroOrOneHotConstraint(variables=set([a, b])))
    c.append(ZeroOrOneHotConstraint(variables=a))
    c.append(ZeroOrOneHotConstraint(label="my label"))
    c.append(ZeroOrOneHotConstraint(strength=20))
    c.append("another type")

    for i in range(len(c) - 1):
        for j in range(i + 1, len(c)):
            assert c[i] != c[j]


def test_zero_or_one_hot_constraint_repr():
    c = ZeroOrOneHotConstraint()
    assert isinstance(c.__repr__(), str)
    assert "ZeroOrOneHotConstraint({" in c.__repr__()
    assert "'constraint_class':" in c.__repr__()
    assert "'variables'" in c.__repr__()
    assert "'label'" in c.__repr__()
    assert "'strength':" in c.__repr__()


def test_zero_or_one_hot_constraint_str():
    c = ZeroOrOneHotConstraint()
    assert isinstance(c.__str__(), str)
    assert "'constraint_class':" in c.__str__()
    assert "'variables'" in c.__str__()
    assert "'label'" in c.__str__()
    assert "'strength':" in c.__str__()
