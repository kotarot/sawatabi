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


def current_time():
    return time.time()


def current_time_as_int():
    return int(time.time())


def current_time_ms():
    return time.time() * 1_000


def current_time_ms_as_int():
    return int(time.time() * 1_000)


def current_time_us():
    return time.time() * 1_000_000


def current_time_us_as_int():
    return int(time.time() * 1_000_000)


def current_time_ns():
    return time.time() * 1_000_000_000


def current_time_ns_as_int():
    return int(time.time() * 1_000_000_000)
