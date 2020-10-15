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

from sawatabi import LogicalModel

################################
# Model
################################

@pytest.mark.parametrize('type', ['ising', 'qubo'])
def test_logical_model_constructor(type):
    model = LogicalModel(type=type)
    assert model.get_type() == type

def test_logical_model_invalid_type():
    with pytest.raises(ValueError):
        model = LogicalModel()

    with pytest.raises(ValueError):
        model = LogicalModel(type='othertype')

    with pytest.raises(ValueError):
        model = LogicalModel(type=12345)

################################
# Array
################################

def test_logical_model_array():
    model = LogicalModel(type='ising')
    model.array('x', shape=(2, 3))

@pytest.mark.parametrize('name,shape', [
    (12345, (2, 3)),
    ('x', 12345),
    ('x', ()),
    ('x', ('a', 'b')),
])
def test_logical_model_invalid_arrays(name, shape):
    model = LogicalModel(type='ising')
    with pytest.raises(TypeError):
        model.array(name, shape=shape)

def test_logical_model_arrays_from_pyqubo():
    import pyqubo
    x = pyqubo.Array.create('x', shape=(2, 3), vartype='BINARY')
    model = LogicalModel(type='qubo')
    model.array(x)

################################
# Add
################################

def test_logical_model_add():
    model = LogicalModel(type='ising')

    with pytest.raises(NotImplementedError):
        model.add_variable()

    with pytest.raises(NotImplementedError):
        model.add_variables()

    with pytest.raises(NotImplementedError):
        model.add_interaction()

    with pytest.raises(NotImplementedError):
        model.add_interactions()
