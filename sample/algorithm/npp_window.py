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
import datetime
import os
from typing import List, Union

import dimod

import sawatabi


def npp_mapping(
    prev_model: sawatabi.model.LogicalModel, prev_sampleset: dimod.SampleSet, elements: List, incoming: List, outgoing: List
) -> sawatabi.model.LogicalModel:
    """
    Mapping -- Update the model based on the input data elements
    """

    model = prev_model
    if len(incoming) > 0:
        # Max index of the incoming elements
        max_index = max([i[1][0] for i in incoming])
        # Get current array size
        x_size = model.get_all_size()
        # Update variables
        x = model.append(name="x", shape=(max_index - x_size + 1,))
    else:
        x = model.get_variables_by_name(name="x")

    # print("x:", x)
    # print("elements:", elements)
    # print("incoming:", incoming)
    # print("outgoing:", outgoing)
    for i in incoming:
        for j in elements:
            if i[0] > j[0]:
                idx_i = i[1][0]
                idx_j = j[1][0]
                coeff = -1.0 * i[1][1] * j[1][1]
                model.add_interaction(
                    target=(x[idx_i], x[idx_j]),
                    coefficient=coeff,
                    attributes={"n": str(j), "attn_ts": j[0]},  # metadata for affected number and timestamp for attenuation
                )

    for o in outgoing:
        idx = o[1][0]
        model.delete_variable(target=x[idx])

    return model


def npp_unmapping(sampleset: dimod.SampleSet, elements: List, incoming: List, outgoing: List) -> str:
    """
    Unmapping -- Decode spins to a problem solution
    """

    outputs = []
    outputs.append("")
    outputs.append("INPUT -->")
    outputs.append(f"  {[e[1][1] for e in elements]}")
    outputs.append(f"  (length: {len(elements)})")
    outputs.append("SOLUTION ==>")

    # Decode spins to solution
    spins = sampleset.samples()[0]

    set_p, set_n = [], []
    n_set_p = n_set_n = 0
    for e in elements:
        if spins[f"x[{e[1][0]}]"] == 1:
            set_p.append(e[1][1])
            n_set_p += e[1][1]
        elif spins[f"x[{e[1][0]}]"] == -1:
            set_n.append(e[1][1])
            n_set_n += e[1][1]
    outputs.append(f"  Set(+) : sum={n_set_p}, elements={set_p}")
    outputs.append(f"  Set(-) : sum={n_set_n}, elements={set_n}")
    outputs.append(f"  diff   : {abs(n_set_p - n_set_n)}")

    return "\n".join(outputs)


def npp_solving(
    solver: Union[sawatabi.solver.LocalSolver, sawatabi.solver.DWaveSolver, sawatabi.solver.OptiganSolver, sawatabi.solver.SawatabiSolver],
    model: sawatabi.model.LogicalModel,
    prev_sampleset: dimod.SampleSet,
    elements: List,
    incoming: List,
    outgoing: List,
) -> dimod.SampleSet:
    """
    Solving -- Solve model and find results (sampleset)
    """

    # Solver options as a dict
    SOLVER_OPTIONS = {
        "num_reads": 1,
        "num_sweeps": 10000,
        "seed": 12345,
    }
    # The main solve.
    physical_model = model.to_physical()
    sampleset = solver.solve(physical_model, **SOLVER_OPTIONS)

    # Set a fallback solver if needed here.
    pass

    return sampleset


def npp_window(
    project: str = None,
    input_path: str = None,
    input_topic: str = None,
    input_subscription: str = None,
    output_path: str = None,
    output_topic: str = None,
    dataflow: bool = False,
    dataflow_bucket: str = None,
) -> None:

    if dataflow and dataflow_bucket:
        yymmddhhmmss = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        pipeline_args = [
            "--runner=DataflowRunner",
            f"--project={project}",
            "--region=asia-northeast1",
            f"--temp_location=gs://{dataflow_bucket}/temp",
            f"--setup_file={os.path.dirname(os.path.abspath(__file__))}/../../setup.py",
            f"--job_name=beamapp-npp-{yymmddhhmmss}",
            # Reference: https://stackoverflow.com/questions/56403572/no-userstate-context-is-available-google-cloud-dataflow
            "--experiments=use_runner_v2",
            # Worker options
            "--autoscaling_algorithm=NONE",
            "--num_workers=1",
            "--max_num_workers=1",
        ]
    else:
        pipeline_args = ["--runner=DirectRunner"]
    # pipeline_args.append("--save_main_session")  # If save_main_session is true, pickle of the session fails on Windows unit tests

    if (project is not None) and ((input_topic is not None) or (input_subscription is not None)):
        pipeline_args.append("--streaming")

    algorithm_options = {
        "window.size": 30,  # required
        "window.period": 5,  # required
        "output.with_timestamp": True,  # optional
        "output.prefix": "<<<\n",  # optional
        "output.suffix": "\n>>>\n",  # optional
    }

    if (project is not None) and (input_topic is not None):
        input_fn = sawatabi.algorithm.IO.read_from_pubsub_as_number(project=project, topic=input_topic)
    elif (project is not None) and (input_subscription is not None):
        input_fn = sawatabi.algorithm.IO.read_from_pubsub_as_number(project=project, subscription=input_subscription)
    elif input_path is not None:
        input_fn = sawatabi.algorithm.IO.read_from_text_as_number(path=input_path)
        algorithm_options["input.reassign_timestamp"] = True

    if output_path is not None:
        output_fn = sawatabi.algorithm.IO.write_to_text(path=output_path)
    elif (project is not None) and (output_topic is not None):
        output_fn = sawatabi.algorithm.IO.write_to_pubsub(project=project, topic=output_topic)
    else:
        output_fn = sawatabi.algorithm.IO.write_to_stdout()

    # Pipeline creation with Sawatabi
    pipeline = sawatabi.algorithm.Window.create_pipeline(
        algorithm_options=algorithm_options,
        input_fn=input_fn,
        map_fn=npp_mapping,
        solve_fn=npp_solving,
        unmap_fn=npp_unmapping,
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
    parser.add_argument("--output-topic", dest="output_topic", help="Google Cloud Pub/Sub topic name to publish the result output to.")
    parser.add_argument(
        "--dataflow",
        dest="dataflow",
        action="store_true",
        help="If true, the application will run on Google Cloud Dataflow (DataflowRunner). If false, it will run on local (DirectRunner).",
    )
    parser.add_argument(
        "--dataflow-bucket",
        dest="dataflow_bucket",
        help="GCS bucket name for temporary files, if the application runs on Google Cloud Dataflow.",
    )
    args = parser.parse_args()

    npp_window(args.project, args.input, args.input_topic, args.input_subscription, args.output, args.output_topic, args.dataflow, args.dataflow_bucket)


if __name__ == "__main__":
    main()
