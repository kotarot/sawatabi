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

from sawatabi.model import PhysicalModel


@pytest.fixture
def model():
    return PhysicalModel(type="ising")


################################
# Physical Model
################################


@pytest.mark.parametrize("type", ["ising", "qubo"])
def test_physical_model_constructor(type):
    model = PhysicalModel(type=type)
    assert model.get_type() == type
    assert model._type == type


################################
# Built-in functions
################################


def test_physical_model_repr(model):
    assert isinstance(model.__repr__(), str)
    assert "PhysicalModel({" in model.__repr__()
    assert "type" in model.__repr__()
    assert "interactions" in model.__repr__()


def test_physical_model_str(model):
    assert isinstance(model.__str__(), str)
    assert "PHYSICAL MODEL" in model.__str__()
    assert "type" in model.__str__()
    assert "interactions" in model.__str__()
