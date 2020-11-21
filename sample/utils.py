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
    print("\n=== version ===\n")
    print("version:", sawatabi.version())
    print("version_info:", sawatabi.version_info())
    print("__version__:", sawatabi.__version__)


def current_time():
    print("\n=== current time ===\n")
    print("sec:", sawatabi.utils.current_time())
    print("ms: ", sawatabi.utils.current_time_ms())
    print("us: ", sawatabi.utils.current_time_us())
    print("ns: ", sawatabi.utils.current_time_ns())


def main():
    version()
    current_time()


if __name__ == "__main__":
    main()
