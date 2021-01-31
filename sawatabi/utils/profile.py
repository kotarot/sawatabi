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

import time
from functools import wraps


def profile(func):
    @wraps(func)
    def profile_time(*args, **kargs):
        start_sec = time.perf_counter()

        func_return = func(*args, **kargs)

        execution_sec = time.perf_counter() - start_sec

        return {
            "return": func_return,
            "profile": {
                "function_name": func.__name__,
                "execution_sec": execution_sec,
            },
        }

    return profile_time
