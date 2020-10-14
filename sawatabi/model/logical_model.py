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

from ..constants import *

class LogicalModel():
    def __init__(self, type=''):
        if type in [MODEL_TYPE_ISING, MODEL_TYPE_QUBO]:
            self.type = type
        else:
            raise ValueError("type must be one of {}.".format([MODEL_TYPE_ISING, MODEL_TYPE_QUBO]))

    def add_variable(self):
        raise NotImplementedError

    def add_interaction(self):
        raise NotImplementedError
