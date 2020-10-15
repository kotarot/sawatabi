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

def test_logical_model_ising():
    model = LogicalModel(type='ising')
    print(model)

def test_logical_model_qubo():
    model = LogicalModel(type='qubo')
    print(model)

def test_logical_model_with_invalid_type():
    with pytest.raises(ValueError):
        model = LogicalModel()

    with pytest.raises(ValueError):
        model = LogicalModel(type='othertype')

    with pytest.raises(ValueError):
        model = LogicalModel(type=12345)
