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

import sawatabi

def version():
    print('version:', sawatabi.utils.version())
    print('version_info:', sawatabi.utils.version_info())

def model():
    model = sawatabi.model.LogicalModel(type='ising')
    a = model.array('x', shape=(2, 3))
    print(model)
    print('--- Returns ---')
    print(a)

if __name__ == '__main__':
    version()
    model()
