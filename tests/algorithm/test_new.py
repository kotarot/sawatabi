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

import pytest

from sample.algorithm import npp_functions
from sawatabi.algorithm import IO, New


def test_new_algorithm_npp_100(capfd):
    algorithm_options = {
        "window.size": 10,
        "output.with_timestamp": True,
        "output.prefix": "<< prefix <<\n",
        "output.suffix": "\n>> suffix >>\n",
        "input.reassign_timestamp": True,
    }

    pipeline_args = ["--runner=DirectRunner"]
    # pipeline_args.append("--save_main_session")  # If save_main_session is true, pickle of the session fails on Windows unit tests

    pipeline = New.create_pipeline(
        algorithm_options=algorithm_options,
        input_fn=IO.read_from_text_as_number(path="tests/algorithm/numbers_100.txt"),
        map_fn=npp_functions.npp_mapping,
        solve_fn=npp_functions.npp_solving,
        unmap_fn=npp_functions.npp_unmapping,
        output_fn=IO.write_to_stdout(),
        pipeline_args=pipeline_args,
    )

    with pytest.warns(UserWarning):
        # Run the pipeline
        result = pipeline.run()  # noqa: F841
        # result.wait_until_finish()

    out, err = capfd.readouterr()

    # Timestamp
    assert "[1970-01-01 00:00:29.999000]" in out
    for i in range(10):
        ts = (i + 1) * 10 - 0.001
        assert datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f%z") in out

    # Check (Count) Solution
    assert out.count("INPUT -->") == 10
    assert out.count("SOLUTION ==>") == 10
    assert out.count("The received event is outdated") == 0
    assert "diff   : 0" in out

    # Output prefix/suffix
    assert out.count("<< prefix <<") == 10
    assert out.count(">> suffix >>") == 10


def test_new_algorithm_repr():
    assert str(New()) == "New()"
