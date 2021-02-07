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
import logging
import re

import apache_beam as beam
from apache_beam import coders
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.userstate import BagStateSpec
from beam_trial import IndexAssigningStatefulDoFn, WithTimestampStrFn, WithTimestampTupleFn

import sawatabi

"""
This script is a trial for implementing sawatabi algorithms, especially aims to investigate interfaces.
Based on: `sample/trial/beam_sawatabi_npp.py`

Sample Usage:
$ GOOGLE_APPLICATION_CREDENTIALS="./gcp-key.json" python sample/trial/beam_sawatabi_npp_for_impl.py --project=your-project --topic=your-topic
$ GOOGLE_APPLICATION_CREDENTIALS="./gcp-key.json" python sample/trial/beam_sawatabi_npp_for_impl.py --project=your-project --subscription=your-subsctiption
"""


################################################################
# User-defined functions and codes
################################################################

def user_code():

    def my_mapping(prev_model, elements, incoming, outgoing):
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

        #print("x:", x)
        #print("elements:", elements)
        #print("incoming:", incoming)
        #print("outgoing:", outgoing)
        for i in incoming:
            for j in elements:
                if i[0] > j[0]:
                    idx_i = i[1][0]
                    idx_j = j[1][0]
                    coeff = -1.0 * i[1][1] * j[1][1]
                    model.add_interaction(target=(x[idx_i], x[idx_j]), coefficient=coeff)

        for o in outgoing:
            idx = o[1][0]
            model.delete_variable(target=x[idx])

        return model


    def my_unmapping(sampleset, elements, incoming, outgoing):
        """
        Unmapping -- Decode spins to a problem solution
        """

        outputs = []
        outputs.append("")
        outputs.append("INPUT -->")
        outputs.append("  " + str([e[1][1] for e in elements]))
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

        outputs.append("")
        return "\n".join(outputs)


    def my_solving(physical_model, elements, incoming, outgoing):
        # Solver instance
        # - LocalSolver
        solver = sawatabi.solver.LocalSolver(exact=False)
        # Solver options as a dict
        SOLVER_OPTIONS = {
            "num_reads": 1,
            "num_sweeps": 10000,
            "seed": 12345,
        }
        # The main solve.
        sampleset = solver.solve(physical_model, **SOLVER_OPTIONS)

        # Set a fallback solver if needed here.
        pass

        return sampleset


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        dest="project",
        required=True,
        help="Google Cloud Pub/Sub project name.")
    parser.add_argument(
        "--topic",
        dest="topic",
        help="Google Cloud Pub/Sub topic name to subscribe messages from.")
    parser.add_argument(
        "--subscription",
        dest="subscription",
        help="Google Cloud Pub/Sub subscription name.")
    args = parser.parse_args()

    if args.topic is not None:
        input_fn = default_input_pubsub_number(project=args.project, topic=args.topic)
    elif args.subscription is not None:
        input_fn = default_input_pubsub_number(project=args.project, subscription=args.subscription)

    # Pipeline creation with Sawatabi
    pipeline = create_pipeline(
        algorithm="windowing",
        algorithm_options={"window_size": 30, "window_period": 5},
        input_fn=input_fn,
        map_fn=my_mapping,
        solve_fn=my_solving,
        unmap_fn=my_unmapping,
        output_fn=default_output_print(),
    )

    # Run the pipeline
    result = pipeline.run()
    result.wait_until_finish()


################################################################
# Assume following functions are prepared in the library
################################################################


class SolveDoFn(beam.DoFn):
    PREV_ELEMENTS = BagStateSpec(name="elements_state", coder=coders.PickleCoder())
    PREV_MODEL = BagStateSpec(name="model_state", coder=coders.PickleCoder())

    def process(self, value, elements_state=beam.DoFn.StateParam(PREV_ELEMENTS), model_state=beam.DoFn.StateParam(PREV_MODEL), map_fn=None, unmap_fn=None, solve_fn=None):
        _, elements = value

        # Sort with the event time.
        # If we sort a list of tuples, the first element of the tuple is recognized as a key by default,
        # so just `sorted` is enough.
        sorted_elements = sorted(elements)

        # generator into a list
        elements_state_as_list = list(elements_state.read())
        model_state_as_list = list(model_state.read())
        # Clear the BagState so we can hold only the latest state
        elements_state.clear()
        model_state.clear()

        # Extract elements from state
        if len(elements_state_as_list) == 0:
            prev_elements = []
        else:
            prev_elements = elements_state_as_list[-1]

        # Resolve outgoing elements in this iteration
        def resolve_outgoing(prev_elements, sorted_elements):
            outgoing = []
            for p in prev_elements:
                if p[0] >= sorted_elements[0][0]:
                    break
                outgoing.append(p)
            return outgoing

        outgoing = resolve_outgoing(prev_elements, sorted_elements)

        # Resolve incoming elements in this iteration
        def resolve_incoming(prev_elements, sorted_elements):
            incoming = []
            if len(prev_elements) == 0:
                incoming = sorted_elements
            else:
                for v in reversed(sorted_elements):
                    if v[0] <= prev_elements[-1][0]:
                        break
                    incoming.insert(0, v)
            return incoming

        incoming = resolve_incoming(prev_elements, sorted_elements)

        # Extract model from state
        if len(model_state_as_list) == 0:
            prev_model = sawatabi.model.LogicalModel(mtype="ising")
        else:
            prev_model = model_state_as_list[-1]

        # Map problem input to the model
        model = map_fn(prev_model, elements, incoming, outgoing)
        physical = model.to_physical()

        #print(model)

        # Register new elements and models to the states
        elements_state.add(sorted_elements)
        model_state.add(model)

        # Solve and unmap to the solution
        try:
            sampleset = solve_fn(physical, elements, incoming, outgoing)
        except Exception as e:
            yield f"Failed to solve: {e}"
        else:
            yield unmap_fn(sampleset, elements, incoming, outgoing)



def default_input_pubsub(project, topic=None, subscription=None):
    if topic is not None:
        messages = beam.io.ReadFromPubSub(topic=f"projects/{project}/topics/{topic}")
    elif subscription is not None:
        messages = beam.io.ReadFromPubSub(subscription=f"projects/{project}/subscriptions/{subscription}")

    return (messages
        | "Decode" >> beam.Map(lambda m: m.decode("utf-8")))


def default_input_pubsub_number(project, topic=None, subscription=None):
    number_pattern = re.compile(r"^[0-9]+$")
    messages = default_input_pubsub(project=project, topic=topic, subscription=subscription)
    return (messages
        | "Filter" >> beam.Filter(lambda element: number_pattern.match(element))
        | "To int" >> beam.Map(lambda e: int(e)))


def default_output_print():
    return beam.Map(print)


def create_pipeline(algorithm, algorithm_options, input_fn=None, map_fn=None, solve_fn=None, unmap_fn=None, output_fn=None):

    pipeline_args = []
    pipeline_args.extend([
        "--runner=DirectRunner",
    ])

    pipeline_options = PipelineOptions(pipeline_args, streaming=True, save_main_session=True)
    p = beam.Pipeline(options=pipeline_options)

    if input_fn is not None:
        inputs = (p
            | "Input" >> input_fn)
    else:
        inputs = p

    elements = (inputs
        | "Prepare key" >> beam.Map(lambda x: (None, x))
        | "Assign global index for Ising variables" >> beam.ParDo(IndexAssigningStatefulDoFn()))

    windows = (elements
        | "Sliding windows" >> beam.WindowInto(beam.window.SlidingWindows(size=algorithm_options["window_size"], period=algorithm_options["window_period"]))
        | "Add timestamp tuple for diff detection" >> beam.ParDo(WithTimestampTupleFn())
        | "Sliding Windows to list" >> beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
        | "Global Window for sliding windows" >> beam.WindowInto(beam.window.GlobalWindows()))

    solved = (windows
        | beam.Map(lambda x: (None, x))
        | "Solve" >> beam.ParDo(SolveDoFn(), map_fn=map_fn, unmap_fn=unmap_fn, solve_fn=solve_fn))

    with_timestamp = (solved
        | "With timestamp for sliding windows" >> beam.ParDo(WithTimestampStrFn()))

    if output_fn is not None:
        outputs = (with_timestamp
            | "Output" >> output_fn)

    return p


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # For library implementation
    user_code()
