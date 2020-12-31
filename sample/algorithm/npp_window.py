#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import argparse

import npp_functions

import sawatabi


def npp_window(project=None, topic=None, subscription=None, input_path=None, output_path=None):

    pipeline_args = ["--runner=DirectRunner"]
    # pipeline_args.append("--save_main_session")  # If save_main_session is true, pickle of the session fails on Windows unit tests
    if project is not None:
        pipeline_args.append("--streaming")

    algorithm_options = {"window.size": 30, "window.period": 5, "output.with_timestamp": True, "output.prefix": "<<<\n", "output.suffix": "\n>>>\n"}

    if topic is not None:
        input_fn = sawatabi.algorithm.IO.read_from_pubsub_as_number(project=project, topic=topic)
    elif subscription is not None:
        input_fn = sawatabi.algorithm.IO.read_from_pubsub_as_number(project=project, subscription=subscription)
    elif input_path is not None:
        input_fn = sawatabi.algorithm.IO.read_from_text_as_number(path=input_path)
        algorithm_options["input.reassign_timestamp"] = True

    if output_path is not None:
        output_fn = sawatabi.algorithm.IO.write_to_text(path=output_path)
    else:
        output_fn = sawatabi.algorithm.IO.write_to_stdout()

    # Pipeline creation with Sawatabi
    pipeline = sawatabi.algorithm.Window.create_pipeline(
        algorithm_options=algorithm_options,
        input_fn=input_fn,
        map_fn=npp_functions.npp_mapping,
        solve_fn=npp_functions.npp_solving,
        unmap_fn=npp_functions.npp_unmapping,
        output_fn=output_fn,
        pipeline_args=pipeline_args,
    )

    # Run the pipeline
    result = pipeline.run()
    result.wait_until_finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", dest="project", help="Google Cloud Pub/Sub project name.")
    parser.add_argument("--topic", dest="topic", help="Google Cloud Pub/Sub topic name to subscribe messages from.")
    parser.add_argument("--subscription", dest="subscription", help="Google Cloud Pub/Sub subscription name.")
    parser.add_argument("--input", dest="input", help="Path to the local file or the GCS object to read from.")
    parser.add_argument("--output", dest="output", help="Path (prefix) to the output file or the object to write to.")
    args = parser.parse_args()

    npp_window(args.project, args.topic, args.subscription, args.input, args.output)


if __name__ == "__main__":
    main()