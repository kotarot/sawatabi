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
import pytest

import apache_beam as beam

from sample.algorithm import npp_functions
from sawatabi.algorithm import IO, Window


def test_window_algorithm_npp_100(capfd):
    algorithm_options = {"window.size": 30, "window.period": 5, "output.with_timestamp": True, "output.prefix": "<<<\n", "output.suffix": "\n>>>\n", "input.reassign_timestamp": True}

    pipeline_args = ["--runner=DirectRunner"]
    # pipeline_args.append("--save_main_session")  # If save_main_session is true, pickle of the session fails on Windows unit tests

    pipeline = Window.create_pipeline(
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
        result = pipeline.run()
        # result.wait_until_finish()

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
    output_path = "tests/algorithm/output.txt"

    algorithm_options = {"window.size": 30, "window.period": 5, "output.with_timestamp": True, "input.reassign_timestamp": True}

    pipeline_args = ["--runner=DirectRunner"]
    # pipeline_args.append("--save_main_session")  # If save_main_session is true, pickle of the session fails on Windows unit tests

    pipeline = Window.create_pipeline(
        algorithm_options=algorithm_options,
        input_fn=IO.read_from_text_as_number(path="tests/algorithm/numbers_10.txt"),
        map_fn=npp_functions.npp_mapping,
        solve_fn=npp_functions.npp_solving,
        unmap_fn=npp_functions.npp_unmapping,
        output_fn=IO.write_to_text(path=output_path),
        pipeline_args=pipeline_args,
    )

    with pytest.warns(UserWarning):
        # Run the pipeline
        result = pipeline.run()
        # result.wait_until_finish()


    assert os.path.exists(f"{output_path}-00000-of-00001")
    os.remove(f"{output_path}-00000-of-00001")


def test_window_algorithm_npp_gcp_and_custom_fn(capfd):
    algorithm_options = {"window.size": 30, "window.period": 10, "input.reassign_timestamp": True}

    input_fn = beam.io.ReadFromText("gs://sawatabi-bucket/numbers_100.txt") | beam.Map(lambda x: int(x))
    output_fn = beam.Map(lambda x: "custom output --- " + x) | beam.Map(print)

    pipeline_args = ["--runner=DirectRunner"]
    # pipeline_args.append("--save_main_session")  # If save_main_session is true, pickle of the session fails on Windows unit tests

    pipeline = Window.create_pipeline(
        algorithm_options=algorithm_options,
        input_fn=input_fn,
        map_fn=npp_functions.npp_mapping,
        solve_fn=npp_functions.npp_solving,
        unmap_fn=npp_functions.npp_unmapping,
        output_fn=output_fn,
        pipeline_args=pipeline_args,
    )

    with pytest.warns(UserWarning):
        # Run the pipeline
        result = pipeline.run()
        # result.wait_until_finish()

    out, err = capfd.readouterr()

    assert out.count("custom output --- ") == 12
    assert "diff   : 0" in out


def test_window_algorithm_npp_map_fails(capfd):
    def invalid_mapping(prev_model, elements, incoming, outgoing):
        raise Exception("Mapping fails!")

    algorithm_options = {"window.size": 30, "window.period": 10, "input.reassign_timestamp": True}

    pipeline_args = ["--runner=DirectRunner"]
    # pipeline_args.append("--save_main_session")  # If save_main_session is true, pickle of the session fails on Windows unit tests

    pipeline = Window.create_pipeline(
        algorithm_options=algorithm_options,
        input_fn=IO.read_from_text_as_number(path="tests/algorithm/numbers_100.txt"),
        map_fn=invalid_mapping,
        solve_fn=npp_functions.npp_solving,
        unmap_fn=npp_functions.npp_unmapping,
        output_fn=IO.write_to_stdout(),
        pipeline_args=pipeline_args,
    )

    # Run the pipeline
    result = pipeline.run()
    # result.wait_until_finish()

    out, err = capfd.readouterr()

    assert out.count("Failed to map: Mapping fails!") == 10
    assert out.count("The received event is outdated") == 2


def test_window_algorithm_npp_unmap_fails(capfd):
    def invalid_unmapping(prev_model, elements, incoming, outgoing):
        raise Exception("Unmapping fails!")

    algorithm_options = {"window.size": 30, "window.period": 10, "input.reassign_timestamp": True}

    pipeline_args = ["--runner=DirectRunner"]
    # pipeline_args.append("--save_main_session")  # If save_main_session is true, pickle of the session fails on Windows unit tests

    pipeline = Window.create_pipeline(
        algorithm_options=algorithm_options,
        input_fn=IO.read_from_text_as_number(path="tests/algorithm/numbers_100.txt"),
        map_fn=npp_functions.npp_mapping,
        solve_fn=npp_functions.npp_solving,
        unmap_fn=invalid_unmapping,
        output_fn=IO.write_to_stdout(),
        pipeline_args=pipeline_args,
    )

    # Run the pipeline
    result = pipeline.run()
    # result.wait_until_finish()

    out, err = capfd.readouterr()

    assert out.count("Failed to unmap: Unmapping fails!") == 10
    assert out.count("The received event is outdated") == 2


def test_window_algorithm_npp_map_fails(capfd):
    def invalid_solving(prev_model, elements, incoming, outgoing):
        raise Exception("Solving fails!")

    algorithm_options = {"window.size": 30, "window.period": 10, "input.reassign_timestamp": True}

    pipeline_args = ["--runner=DirectRunner"]
    # pipeline_args.append("--save_main_session")  # If save_main_session is true, pickle of the session fails on Windows unit tests

    pipeline = Window.create_pipeline(
        algorithm_options=algorithm_options,
        input_fn=IO.read_from_text_as_number(path="tests/algorithm/numbers_100.txt"),
        map_fn=npp_functions.npp_mapping,
        solve_fn=invalid_solving,
        unmap_fn=npp_functions.npp_unmapping,
        output_fn=IO.write_to_stdout(),
        pipeline_args=pipeline_args,
    )

    # Run the pipeline
    result = pipeline.run()
    # result.wait_until_finish()

    out, err = capfd.readouterr()

    assert out.count("Failed to solve: Solving fails!") == 10
    assert out.count("The received event is outdated") == 2


def test_window_algorithm_repr():
    assert str(Window()) == "Window()"
