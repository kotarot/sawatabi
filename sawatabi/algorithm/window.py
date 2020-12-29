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

import apache_beam as beam
from apache_beam import coders
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.userstate import BagStateSpec

import sawatabi
from sawatabi.algorithm.abstract_algorithm import AbstractAlgorithm
from sawatabi.algorithm.io import IO


class Window(AbstractAlgorithm):
    def __init__(self):
        super().__init__()

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
                resultset = solve_fn(physical, elements, incoming, outgoing)
            except Exception as e:
                yield f"Failed to solve: {e}"
            else:
                yield unmap_fn(resultset, elements, incoming, outgoing)

    @staticmethod
    def create_pipeline(algorithm_options, input_fn=None, map_fn=None, solve_fn=None, unmap_fn=None, output_fn=None, pipeline_args=["--runner=DirectRunner"]):
        pipeline_options = PipelineOptions(pipeline_args, streaming=True, save_main_session=True)
        p = beam.Pipeline(options=pipeline_options)

        if input_fn is not None:
            inputs = (p
                | "Input" >> input_fn)
        else:
            inputs = p

        elements = (inputs
            | "Prepare key" >> beam.Map(lambda x: (None, x))
            | "Assign global index for Ising variables" >> beam.ParDo(AbstractAlgorithm.IndexAssigningStatefulDoFn()))

        windows = (elements
            | "Sliding windows" >> beam.WindowInto(beam.window.SlidingWindows(size=algorithm_options["window_size"], period=algorithm_options["window_period"]))
            | "Add timestamp tuple for diff detection" >> beam.ParDo(AbstractAlgorithm.WithTimestampTupleFn())
            | "Sliding Windows to list" >> beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
            | "Global Window for sliding windows" >> beam.WindowInto(beam.window.GlobalWindows()))

        solved = (windows
            | beam.Map(lambda x: (None, x))
            | "Solve" >> beam.ParDo(sawatabi.algorithm.Window.SolveDoFn(), map_fn=map_fn, unmap_fn=unmap_fn, solve_fn=solve_fn))

        with_timestamp = (solved
            | "With timestamp for sliding windows" >> beam.ParDo(AbstractAlgorithm.WithTimestampStrFn()))

        if output_fn is not None:
            outputs = (with_timestamp
                | "Output" >> output_fn)

        return p

    ################################
    # Built-in functions
    ################################

    def __repr__(self):
        return f"Window({self.__str__()})"

    def __str__(self):
        data = {}
        return str(data)
