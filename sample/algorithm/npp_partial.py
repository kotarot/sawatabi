#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import argparse
from typing import List

import npp_window

import sawatabi


def npp_partial(project: str = None, input_path: str = None, input_topic: str = None, input_subscription: str = None, output_path: str = None) -> None:

    pipeline_args = ["--runner=DirectRunner"]
    # pipeline_args.append("--save_main_session")  # If save_main_session is true, pickle of the session fails on Windows unit tests

    if (project is not None) and ((input_topic is not None) or (input_subscription is not None)):
        pipeline_args.append("--streaming")

    # Filter function for patial algorithm
    def filter_fn(x: List) -> bool:
        # If the number is greater than 90, it remains in the window.
        if x[1][1] > 90:
            return True
        return False

    algorithm_options = {
        "window.size": 10,  # required
        "window.period": 10,  # required
        "filter_fn": filter_fn,  # required
        "output.with_timestamp": True,  # optional
        "output.prefix": "<<<\n",  # optional
        "output.suffix": "\n>>>\n",  # optional
    }

    if input_path is not None:
        input_fn = sawatabi.algorithm.IO.read_from_text_as_number(path=input_path)
        algorithm_options["input.reassign_timestamp"] = True
    elif (project is not None) and (input_topic is not None):
        input_fn = sawatabi.algorithm.IO.read_from_pubsub_as_number(project=project, topic=input_topic)
    elif (project is not None) and (input_subscription is not None):
        input_fn = sawatabi.algorithm.IO.read_from_pubsub_as_number(project=project, subscription=input_subscription)

    if output_path is not None:
        output_fn = sawatabi.algorithm.IO.write_to_text(path=output_path)
    else:
        output_fn = sawatabi.algorithm.IO.write_to_stdout()

    # Pipeline creation with Sawatabi
    pipeline = sawatabi.algorithm.Partial.create_pipeline(
        algorithm_options=algorithm_options,
        input_fn=input_fn,
        map_fn=npp_window.npp_mapping,
        solve_fn=npp_window.npp_solving,
        unmap_fn=npp_window.npp_unmapping,
        output_fn=output_fn,
        solver=sawatabi.solver.LocalSolver(exact=False),  # use LocalSolver
        initial_mtype="ising",
        pipeline_args=pipeline_args,
    )

    # Run the pipeline
    result = pipeline.run()
    result.wait_until_finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", dest="project", help="Google Cloud Platform project name.")
    parser.add_argument("--input", dest="input", help="Path to the local file or the GCS object to read from.")
    parser.add_argument("--input-topic", dest="input_topic", help="Google Cloud Pub/Sub topic name to subscribe messages from.")
    parser.add_argument("--input-subscription", dest="input_subscription", help="Google Cloud Pub/Sub subscription name.")
    parser.add_argument("--output", dest="output", help="Path (prefix) to the output file or the object to write to.")
    args = parser.parse_args()

    npp_partial(args.project, args.input, args.input_topic, args.input_subscription, args.output)


if __name__ == "__main__":
    main()
