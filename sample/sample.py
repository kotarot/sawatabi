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


def sample_version():
    print("\n=== version ===")
    print("version:", sawatabi.version())
    print("version_info:", sawatabi.version_info())


def sample_current_time():
    print("\n=== current time ===")
    print("sec:", sawatabi.utils.current_time())
    print("ms: ", sawatabi.utils.current_time_ms())
    print("us: ", sawatabi.utils.current_time_us())
    print("ns: ", sawatabi.utils.current_time_ns())


def sample_model_1d():
    print("\n=== model ===")
    model = sawatabi.model.LogicalModel(type="ising")
    x = model.array("x", shape=(2,))
    print("--- Model ---")
    print(model)
    print("--- Return Value ---")
    print(x)

    model.update_variable(x[0], coefficient=1.0)
    model.update_variable(x[1], coefficient=2.0, attributes={"foo": "bar"})
    model.update_interaction((x[0], x[1]), coefficient=-3.0)

    x = model.append(shape=(1,))
    print("--- Model ---")
    print(model)
    print("--- Return Value ---")
    print(x)

    model.update_variable(
        x[2],
        coefficient=4.0,
        attributes={"myattr1": "foo", "myattr2": "bar"},
        timestamp=15988860000000,
    )


if __name__ == "__main__":
    sample_version()
    sample_current_time()
    sample_model_1d()
