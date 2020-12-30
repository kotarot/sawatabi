# Copyright 2020 Kotaro Terada
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

import datetime
import os

from sample.algorithm import npp_window


def test_window_algorithm_npp_100(capfd):
    npp_window.npp_window(project=None, topic=None, subscription=None, path="tests/algorithm/numbers_100.txt", output=None)

    out, err = capfd.readouterr()

    # Timestamp
    assert "[1970-01-01 00:00:29.999000]" in out
    for i in range(25):
        ts = (i + 1) * 5 - 0.001
        assert datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f%z") in out

    # Check (Count) Solution
    assert out.count("INPUT -->") == 20
    assert out.count("SOLUTION ==>") == 20
    assert out.count("The received event is outdated") == 5
    assert "diff   : 0" in out

    # Output prefix/suffix
    assert out.count("<<<") == 25
    assert out.count(">>>") == 25


def test_window_algorithm_npp_10():
    output = "tests/algorithm/output.txt"
    npp_window.npp_window(project=None, topic=None, subscription=None, path="tests/algorithm/numbers_10.txt", output=output)

    assert os.path.exists(f"{output}-00000-of-00001")
