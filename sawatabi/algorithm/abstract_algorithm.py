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

import apache_beam as beam
from apache_beam import coders
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.userstate import BagStateSpec, CombiningValueStateSpec

import sawatabi
from sawatabi.base_mixin import BaseMixin


class AbstractAlgorithm(BaseMixin):
    class IndexAssigningStatefulDoFn(beam.DoFn):
        INDEX_STATE = CombiningValueStateSpec(name="index", coder=coders.PickleCoder(), combine_fn=sum)

        def process(self, element, index=beam.DoFn.StateParam(INDEX_STATE)):
            _, value = element
            current_index = index.read()
            index.add(1)
            yield (current_index, value)

    class WithTimestampTupleFn(beam.DoFn):
        def process(self, data, timestamp=beam.DoFn.TimestampParam):
            yield (float(timestamp), data)

    class WithTimestampStrFn(beam.DoFn):
        def process(self, data, timestamp=beam.DoFn.TimestampParam):
            yield f"[{timestamp.to_utc_datetime()}] {data}"

    class SolveDoFn(beam.DoFn):
        PREV_TIMESTAMP = BagStateSpec(name="timestamp_state", coder=coders.PickleCoder())
        PREV_ELEMENTS = BagStateSpec(name="elements_state", coder=coders.PickleCoder())
        PREV_MODEL = BagStateSpec(name="model_state", coder=coders.PickleCoder())

        def process(
            self,
            value,
            timestamp=beam.DoFn.TimestampParam,
            timestamp_state=beam.DoFn.StateParam(PREV_TIMESTAMP),
            elements_state=beam.DoFn.StateParam(PREV_ELEMENTS),
            model_state=beam.DoFn.StateParam(PREV_MODEL),
            algorithm=None,
            map_fn=None,
            unmap_fn=None,
            solve_fn=None,
        ):
            _, elements = value

            # Sort with the event time.
            # If we sort a list of tuples, the first element of the tuple is recognized as a key by default,
            # so just `sorted` is enough.
            sorted_elements = sorted(elements)

            # generator into a list
            timestamp_state_as_list = list(timestamp_state.read())
            elements_state_as_list = list(elements_state.read())
            model_state_as_list = list(model_state.read())

            # Extract the previous timestamp, elements, and model from state
            if len(timestamp_state_as_list) == 0:
                prev_timestamp = -1.0
            else:
                prev_timestamp = timestamp_state_as_list[-1]
            if len(elements_state_as_list) == 0:
                prev_elements = []
            else:
                prev_elements = elements_state_as_list[-1]
            if len(model_state_as_list) == 0:
                prev_model = sawatabi.model.LogicalModel(mtype="ising")
            else:
                prev_model = model_state_as_list[-1]

            # Sometimes, when we use the sliding window algorithm for a bounded data (such as a local file),
            # we may receive an outdated event whose timestamp is older than timestamp of previously processed event.
            if float(timestamp) < float(prev_timestamp):
                yield (
                    f"The received event is outdated: Timestamp is {timestamp.to_utc_datetime()}, "
                    + f"while an event with timestamp of {timestamp.to_utc_datetime()} has been already processed."
                )
                return

            if algorithm == sawatabi.constants.ALGORITHM_INCREMENTAL:
                sorted_elements.extend(prev_elements)
                sorted_elements = sorted(sorted_elements)

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

            # Clear the BagState so we can hold only the latest state, and
            # Register new timestamp and elements to the states
            timestamp_state.clear()
            timestamp_state.add(timestamp)
            elements_state.clear()
            elements_state.add(sorted_elements)

            # Map problem input to the model
            try:
                model = map_fn(prev_model, sorted_elements, incoming, outgoing)
                physical = model.to_physical()
            except Exception as e:
                yield f"Failed to map: {e}"
                return

            # Clear the BagState so we can hold only the latest state, and
            # Register new model to the state
            model_state.clear()
            model_state.add(model)

            # Solve and unmap to the solution
            try:
                resultset = solve_fn(physical, sorted_elements, incoming, outgoing)
            except Exception as e:
                yield f"Failed to solve: {e}"
                return

            try:
                yield unmap_fn(resultset, sorted_elements, incoming, outgoing)
            except Exception as e:
                yield f"Failed to unmap: {e}"

    @classmethod
    def _create_pipeline(
        cls,
        algorithm,
        algorithm_transform,
        algorithm_options,
        input_fn=None,
        map_fn=None,
        solve_fn=None,
        unmap_fn=None,
        output_fn=None,
        pipeline_args=["--runner=DirectRunner"],
    ):
        pipeline_options = PipelineOptions(pipeline_args)
        p = beam.Pipeline(options=pipeline_options)

        # fmt: off

        # --------------------------------
        # Input part
        # --------------------------------

        inputs = p
        if input_fn is not None:
            inputs = (p
                | "Input" >> input_fn)

        with_indices = (inputs
            | "Prepare key" >> beam.Map(lambda element: (None, element))
            | "Assign global index for Ising variables" >> beam.ParDo(AbstractAlgorithm.IndexAssigningStatefulDoFn()))

        if "input.reassign_timestamp" in algorithm_options:
            # Add (Re-assign) event timestamp based on the index
            # - element[0]: index
            # - element[1]: data
            with_indices = (with_indices
                | "Assign timestamp by index" >> beam.Map(lambda element: beam.window.TimestampedValue(element, element[0])))

        # --------------------------------
        # Algorithm part
        # --------------------------------

        algorithm_transformed = with_indices | algorithm_transform

        # --------------------------------
        # Solving part
        # --------------------------------

        solved = (algorithm_transformed
            | "Make windows to key-value pairs for stateful DoFn" >> beam.Map(lambda element: (None, element))
            | "Solve" >> beam.ParDo(sawatabi.algorithm.Window.SolveDoFn(), algorithm=algorithm, map_fn=map_fn, unmap_fn=unmap_fn, solve_fn=solve_fn))

        # --------------------------------
        # Output part
        # --------------------------------

        if "output.with_timestamp" in algorithm_options:
            solved = (solved
                | "With timestamp for each window" >> beam.ParDo(AbstractAlgorithm.WithTimestampStrFn()))

        if "output.prefix" in algorithm_options:
            solved = (solved
                | "Add output prefix" >> beam.Map(lambda element: algorithm_options["output.prefix"] + element))
        if "output.suffix" in algorithm_options:
            solved = (solved
                | "Add output suffix" >> beam.Map(lambda element: element + algorithm_options["output.suffix"]))

        if output_fn is not None:
            outputs = (solved  # noqa: F841
                | "Output" >> output_fn)

        # fmt: on

        return p
