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

import time

import numpy

import sawatabi


def test_current_time():
    current_time = time.time()

    # Check the time approximately

    # sec
    numpy.testing.assert_approx_equal(
        actual=sawatabi.utils.current_time(), desired=current_time, significant=9
    )

    # milli sec
    numpy.testing.assert_approx_equal(
        actual=sawatabi.utils.current_time_ms(),
        desired=current_time * 1_000,
        significant=9,
    )

    # micro sec
    numpy.testing.assert_approx_equal(
        actual=sawatabi.utils.current_time_us(),
        desired=current_time * 1_000_000,
        significant=9,
    )

    # nano sec
    numpy.testing.assert_approx_equal(
        actual=sawatabi.utils.current_time_ns(),
        desired=current_time * 1_000_000_000,
        significant=9,
    )
