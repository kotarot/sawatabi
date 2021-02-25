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

import datetime
from typing import List

import pytest

from sample.algorithm import npp_window
from sawatabi.algorithm import IO, Partial


def test_partial_algorithm_npp_100(capfd):
    # Filter function for patial algorithm
    def filter_fn(x: List) -> bool:
        # If the number is greater than 90, it remains in the window.
        if x[1][1] > 90:
            return True
        return False

    algorithm_options = {
        "window.size": 10,
        "window.period": 10,
        "filter_fn": filter_fn,
        "output.with_timestamp": True,
        "output.prefix": "<< prefix <<\n",
        "output.suffix": "\n>> suffix >>\n",
        "input.reassign_timestamp": True,
    }

    pipeline_args = ["--runner=DirectRunner"]
    # pipeline_args.append("--save_main_session")  # If save_main_session is true, pickle of the session fails on Windows unit tests

    pipeline = Partial.create_pipeline(
        algorithm_options=algorithm_options,
        input_fn=IO.read_from_text_as_number(path="tests/algorithm/numbers_100.txt"),
        map_fn=npp_window.npp_mapping,
        solve_fn=npp_window.npp_solving,
        unmap_fn=npp_window.npp_unmapping,
        output_fn=IO.write_to_stdout(),
        pipeline_args=pipeline_args,
    )

    with pytest.warns(UserWarning):
        # Run the pipeline
        result = pipeline.run()  # noqa: F841
        # result.wait_until_finish()

    out, err = capfd.readouterr()

    # Timestamp
    for i in range(10):
        ts = (i + 1) * 10 - 0.001
        assert datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f%z") in out

    # Check inputs
    assert "[47, 60, 87, 60, 91, 71, 28, 37, 7, 65]" in out  # 1--10
    assert "[91, 28, 29, 38, 55, 6, 75, 57, 49, 34, 83]" in out  # 11--20 + over 90
    assert "[91, 30, 46, 78, 29, 99, 32, 86, 82, 7, 81]" in out  # 21--30 + over 90
    assert "[91, 99, 90, 12, 20, 65, 42, 20, 47, 7, 52, 78]" in out  # 31--40 + over 90
    assert "[91, 99, 2, 74, 51, 75, 26, 7, 36, 46, 65, 29]" in out  # 41--50 + over 90
    assert "[91, 99, 20, 27, 76, 55, 68, 93, 9, 67, 78, 19]" in out  # 51--60 + over 90
    assert "[91, 99, 93, 23, 76, 63, 81, 5, 22, 99, 79, 44, 99]" in out  # 61--70 + over 90
    assert "[91, 99, 93, 99, 99, 2, 61, 14, 37, 1, 62, 64, 62, 30, 6]" in out  # 71--80 + over 90
    assert "[91, 99, 93, 99, 99, 27, 12, 28, 67, 18, 94, 29, 79, 16, 49]" in out  # 81--90 + over 90
    assert "[91, 99, 93, 99, 99, 94, 87, 95, 6, 42, 41, 23, 73, 67, 74, 27]" in out  # 91--100 + over 100

    # Check (Count) Solution
    assert out.count("INPUT -->") == 10
    assert out.count("SOLUTION ==>") == 10
    assert out.count("The received event is outdated") == 0
    assert "diff   : 5" in out

    # Output prefix/suffix
    assert out.count("<< prefix <<") == 10
    assert out.count(">> suffix >>") == 10


def test_partial_algorithm_repr():
    assert str(Partial()) == "Partial()"
