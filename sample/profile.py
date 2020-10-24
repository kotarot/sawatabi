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

import random

import sawatabi


@sawatabi.utils.profile
def create_n_variable_random_model(n=4, seed=None):
    if seed is not None:
        random.seed(seed)

    model = sawatabi.model.LogicalModel(mtype="ising")
    x = model.variables("x", shape=(n,))
    for i in range(n):
        model.add_interaction(x[i], coefficient=random.random())
    for i in range(n - 1):
        for j in range(i + 1, n):
            model.add_interaction((x[i], x[j]), coefficient=random.random())


def main():
    result = create_n_variable_random_model(n=1000, seed=12345)
    print(result["profile"])


if __name__ == "__main__":
    main()
