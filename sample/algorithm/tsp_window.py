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
import os
from typing import Dict, List, Tuple

import dimod
import pyqubo
from geopy.distance import geodesic

import sawatabi


def tsp_mapping(
    prev_model: sawatabi.model.LogicalModel,
    curr_data: List[Tuple[float, Tuple[int, Dict[str, List[float]]]]],
    incoming: List[Tuple[float, Tuple[int, Dict[str, List[float]]]]],
    outgoing: List[Tuple[float, Tuple[int, Dict[str, List[float]]]]],
) -> sawatabi.model.LogicalModel:
    """
    Mapping -- Update model based on the input data elements

    Parameters
    ----------
    prev_model : sawatabi.model.LogicalModel
        Previous LogicalModel.
    curr_data : List
        Elements in the current window (= prev data + incoming - outgoing).
        curr_data : [
            (1.0, (1, {'Sendai': [140.87194, 38.26889]})),
            (2.0, (2, {'Tokyo': [139.69167, 35.689440000000005]})),
            (3.0, (3, {'Yokohama': [139.6425, 35.44778]})),
            (4.0, (4, {'Nagoya': [136.90667, 35.180279999999996]})),
            (5.0, (5, {'Kyoto': [135.75556, 35.021390000000004]})) ]
    incoming : List
        Elements just coming into the current window.
        incoming: [
            (5.0, (5, {'Kyoto': [135.75556, 35.021390000000004]})) ]
    outgoing : List
        Elements just going from the current window.
        outgoing: [
            (0.0, (0, {'Sapporo': [141.34694, 43.064170000000004]})) ]

    Returns
    -------
    model : sawatabi.model.LogicalModel
        LogicalModel of TSP created from data in the current window.
    """

    model = sawatabi.model.LogicalModel(mtype="qubo")

    # print(f"incoming: {incoming}", type(incoming))
    # print(f"curr_data: {curr_data}", type(curr_data))
    # print(f"outgoing: {outgoing}", type(outgoing))

    # prepare binary vector with bit(i, j)
    n_city = len(curr_data)
    if n_city > 0:
        binary_vector = model.append("city", shape=(n_city, n_city))  # Update variables
    else:
        return model

    # Constraint not to visit more than two cities at the same time.
    time_const = 0.0
    for i in range(n_city):
        time_const += pyqubo.Constraint((pyqubo.Sum(0, n_city, lambda j: binary_vector[i, j]) - 1) ** 2, label="time{}".format(i))

    # Constraint not to visit the same city more than twice.
    city_const = 0.0
    for j in range(n_city):
        city_const += pyqubo.Constraint((pyqubo.Sum(0, n_city, lambda i: binary_vector[i, j]) - 1) ** 2, label="city{}".format(j))

    n_cities = [list(c[1][1].values())[0] for c in curr_data]
    traveling_distance = get_traveling_distance(n_city, n_cities, binary_vector)
    hamiltonian_tsp = traveling_distance + pyqubo.Placeholder("time") * time_const + pyqubo.Placeholder("city") * city_const

    model.from_pyqubo(hamiltonian_tsp)
    # print(f"model: {model}", type(model))

    return model


def get_traveling_distance(n_city: int, n_cities: List, x: pyqubo.Array) -> float:
    distance = 0.0
    for i in range(n_city):  # i: city to visit
        for j in range(n_city):  # j: city to visit
            for k in range(n_city):  # k: visit order
                # Scale down O(100)km -> O(1)km or convenience
                long_lat_dist = geodesic((n_cities[i][1], n_cities[i][0]), (n_cities[j][1], n_cities[j][0])).km / 100
                distance += long_lat_dist * x[k, i] * x[(k + 1) % n_city, j]

    return distance


def tsp_unmapping(sampleset: dimod.SampleSet, elements: List[Tuple[float, Tuple[int, Dict[str, List[float]]]]], incoming: List, outgoing: List) -> str:
    """
    Unmapping -- Decode spins to a problem solution

    Parameters
    ----------
    sampleset : dimod.SampleSet
        Sampling result of the QUBO.
    elements : List
        Elements in the current window.
    incoming : List
        Elements just coming into the current window.
    outgoing : List
        Elements just going from the current window.

    Returns
    -------
    order_to_visit : str
        Solution (order of the cities) as a string.
    """
    outputs = ["", "INPUT -->", "  " + str([e[1][1] for e in elements]), "SOLUTION ==>"]

    # Get order from the solution
    def get_order_to_visit(solution: Dict, elements: List) -> List:
        # store order of the city
        order_to_visit = []
        for i, _ in enumerate(elements):
            for j, e in enumerate(elements):
                if solution[f"city[{i}][{j}]"] == 1:
                    order_to_visit.append(list(e[1][1].keys())[0])
                    break

        return order_to_visit

    return "\n".join(outputs) + "\n  " + " -> ".join(get_order_to_visit(sampleset.samples()[0], elements))


def tsp_solving(model: sawatabi.model.LogicalModel, elements: List, incoming: List, outgoing: List) -> dimod.SampleSet:
    """
    Solving -- Solve model and find results (sampleset)
    """

    from sawatabi.solver import LocalSolver

    # Solver instance
    # - LocalSolver
    solver = LocalSolver(exact=False)
    # Solver options as a dict
    SOLVER_OPTIONS = {
        "num_reads": 1,
        "num_sweeps": 10000,
        "seed": 12345,
    }
    # The main solve.
    physical_model = model.to_physical(placeholder={"time": 17.5, "city": 15.0})
    sampleset = solver.solve(physical_model, **SOLVER_OPTIONS)

    # Set a fallback solver if needed here.
    pass

    return sampleset


def tsp_window(
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
        pipeline_args = [
            "--runner=DataflowRunner",
            f"--project={project}",
            "--region=asia-northeast1",
            f"--temp_location=gs://{dataflow_bucket}/temp",
            f"--setup_file={os.path.dirname(os.path.abspath(__file__))}/../../setup.py",
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

    algorithm_options = {"window.size": 5, "window.period": 1, "output.with_timestamp": True, "output.prefix": "<<<\n", "output.suffix": "\n>>>\n"}

    if (project is not None) and (input_topic is not None):
        input_fn = sawatabi.algorithm.IO.read_from_pubsub_as_number(project=project, topic=input_topic)
    elif (project is not None) and (input_subscription is not None):
        input_fn = sawatabi.algorithm.IO.read_from_pubsub_as_number(project=project, subscription=input_subscription)
    elif input_path is not None:
        input_fn = sawatabi.algorithm.IO.read_from_text_as_json(path=input_path)
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
        map_fn=tsp_mapping,
        solve_fn=tsp_solving,
        unmap_fn=tsp_unmapping,
        output_fn=output_fn,
        initial_mtype="qubo",
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

    tsp_window(args.project, args.input, args.input_topic, args.input_subscription, args.output, args.output_topic, args.dataflow, args.dataflow_bucket)


if __name__ == "__main__":
    main()
