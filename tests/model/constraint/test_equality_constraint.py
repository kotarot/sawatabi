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

from sawatabi.model.constraint import EqualityConstraint

################################
# Equality Constraint
################################


def test_equality_constraint():
    c = EqualityConstraint()
    assert c.get_constraint_class() == "EqualityConstraint"
    assert c.get_variables_1() == set()
    assert c.get_variables_2() == set()
    assert c.get_label() == "Default Equality Constraint"
    assert c.get_strength() == 1.0

    x0 = pyqubo.Spin("x0")
    x1 = pyqubo.Spin("x1")
    y0 = pyqubo.Binary("y0")
    y1 = pyqubo.Binary("y1")

    c.add_variable_to_1(x0)
    assert c.get_variables_1() == set([x0])
    assert c.get_variables_2() == set()

    c.add_variable_to_1([x1])
    assert c.get_variables_1() == set([x0, x1])
    assert c.get_variables_2() == set()

    c.add_variable_to_2(set([y0]))
    assert c.get_variables_1() == set([x0, x1])
    assert c.get_variables_2() == set([y0])

    c.add_variable_to_2(variables=[y0, y1])
    assert c.get_variables_1() == set([x0, x1])
    assert c.get_variables_2() == set([y0, y1])

    c.remove_variable_from_1(variables=x0)
    assert c.get_variables_1() == set([x1])
    assert c.get_variables_2() == set([y0, y1])

    c.remove_variable_from_2(variables=y1)
    assert c.get_variables_1() == set([x1])
    assert c.get_variables_2() == set([y0])

    with pytest.raises(ValueError):
        c.remove_variable_from_1(variables=[x0])

    with pytest.raises(ValueError):
        c.remove_variable_from_2(variables=[y1])


def test_equality_constraint_constructors():
    a = pyqubo.Array.create("a", shape=(2, 2), vartype="SPIN")

    c1 = EqualityConstraint(variables_1=a)
    assert c1.get_variables_1() == set([a[0, 0], a[0, 1], a[1, 0], a[1, 1]])
    assert c1.get_variables_2() == set()
    assert c1.get_label() == "Default Equality Constraint"
    assert c1.get_strength() == 1.0

    c2 = EqualityConstraint(variables_2=a)
    assert c2.get_variables_1() == set()
    assert c2.get_variables_2() == set([a[0, 0], a[0, 1], a[1, 0], a[1, 1]])
    assert c2.get_label() == "Default Equality Constraint"
    assert c2.get_strength() == 1.0

    c3 = EqualityConstraint(label="my label")
    assert c3.get_variables_1() == set()
    assert c3.get_variables_2() == set()
    assert c3.get_label() == "my label"
    assert c3.get_strength() == 1.0

    c4 = EqualityConstraint(strength=20)
    assert c4.get_variables_1() == set()
    assert c4.get_variables_2() == set()
    assert c4.get_label() == "Default Equality Constraint"
    assert c4.get_strength() == 20.0


def test_equality_constraint_typeerror():
    with pytest.raises(TypeError):
        EqualityConstraint(variables_1="invalid type")

    with pytest.raises(TypeError):
        EqualityConstraint(variables_1=set([1, 2, 3]))

    with pytest.raises(TypeError):
        EqualityConstraint(variables_2="invalid type")

    with pytest.raises(TypeError):
        EqualityConstraint(variables_2=set([1, 2, 3]))

    with pytest.raises(TypeError):
        EqualityConstraint(label=12345)

    with pytest.raises(TypeError):
        EqualityConstraint(strength="invalid type")


################################
# Built-in functions
################################


def test_equality_constraint_eq():
    assert EqualityConstraint() == EqualityConstraint()

    a = pyqubo.Spin("a")
    b = pyqubo.Binary("b")
    c1 = EqualityConstraint(variables_1=set([a]), variables_2=b, label="my label", strength=20)
    c2 = EqualityConstraint(variables_1=[a], variables_2=b, label="my label", strength=2 * 10)
    assert c1 == c2


def test_equality_constraint_ne():
    a = pyqubo.Spin("a")
    b = pyqubo.Binary("b")
    c = []
    c.append(EqualityConstraint())
    c.append(EqualityConstraint(variables_1=[a], variables_2=[b], label="my label", strength=20))
    c.append(EqualityConstraint(variables_1=[a], variables_2=[b]))
    c.append(EqualityConstraint(variables_1=set([a, b])))
    c.append(EqualityConstraint(variables_2=set([a, b])))
    c.append(EqualityConstraint(variables_1=a))
    c.append(EqualityConstraint(variables_2=a))
    c.append(EqualityConstraint(variables_1=b))
    c.append(EqualityConstraint(variables_2=b))
    c.append(EqualityConstraint(label="my label"))
    c.append(EqualityConstraint(strength=20))
    c.append("another type")

    for i in range(len(c) - 1):
        for j in range(i + 1, len(c)):
            assert c[i] != c[j]


def test_equality_constraint_repr():
    c = EqualityConstraint()
    assert isinstance(c.__repr__(), str)
    assert "EqualityConstraint({" in c.__repr__()
    assert "'constraint_class':" in c.__repr__()
    assert "'variables_1'" in c.__repr__()
    assert "'variables_2'" in c.__repr__()
    assert "'label'" in c.__repr__()
    assert "'strength':" in c.__repr__()


def test_equality_constraint_str():
    c = EqualityConstraint()
    assert isinstance(c.__str__(), str)
    assert "'constraint_class':" in c.__str__()
    assert "'variables_1'" in c.__str__()
    assert "'variables_2'" in c.__str__()
    assert "'label'" in c.__str__()
    assert "'strength':" in c.__str__()
