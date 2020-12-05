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
from apache_beam.testing.test_stream import TestStream
from apache_beam.transforms.userstate import CombiningValueStateSpec
from apache_beam.transforms.window import TimestampedValue


# Stateful DoFn, based on:
#   - https://beam.apache.org/blog/stateful-processing/
#   - https://github.com/apache/beam/blob/30f9a607509940f78459e4fba847617399780246/sdks/python/apache_beam/transforms/userstate_test.py
class IndexAssigningStatefulDoFn(beam.DoFn):
    INDEX_STATE = CombiningValueStateSpec("index", sum)

    def process(self, element, state=beam.DoFn.StateParam(INDEX_STATE)):
        unused_key, value = element
        current_index = state.read()
        yield (current_index, value)
        state.add(1)


def run():
    events = TestStream().add_elements([
        TimestampedValue(0, 1600000000),
        TimestampedValue(1, 1600000001),
        TimestampedValue(1, 1600000002),
        TimestampedValue(2, 1600000003),
        TimestampedValue(3, 1600000004),
        TimestampedValue(5, 1600000005),
        TimestampedValue(8, 1600000006),
        TimestampedValue(13, 1600000007),
    ])

    with beam.Pipeline() as p:
        # Assign an index for each event.
        #
        # Output:
        #   (0, 0)
        #   (1, 1)
        #   (2, 1)
        #   (3, 2)
        #   (4, 3)
        #   (5, 5)
        #   (6, 8)
        #   (7, 13)
        """
        _ = (p
            | events
            | beam.Map(lambda x: (None, x))
            | beam.ParDo(IndexAssigningStatefulDoFn())
            | beam.Map(print))
        """

        # Try to assign an index for each fixed window (size = 2),
        # resulting in all indices having 0 (stateful not working).
        #
        # Output:
        #   (0, [0, 1])
        #   (0, [1, 2])
        #   (0, [3, 5])
        #   (0, [8, 13])
        """
        _ = (p
            | events
            | beam.WindowInto(beam.window.FixedWindows(size=2))
            | beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
            | beam.Map(lambda x: (None, x))
            | beam.ParDo(IndexAssigningStatefulDoFn())
            | beam.Map(print))
        """

        # We need to apply the fixed windows into a global window before StatefulDoFn.
        #
        # Output:
        #   (0, [0, 1])
        #   (1, [1, 2])
        #   (2, [3, 5])
        #   (3, [8, 13])
        _ = (p
            | events
            | beam.WindowInto(beam.window.FixedWindows(size=2))
            | beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
            | "Global Windows" >> beam.WindowInto(beam.window.GlobalWindows())
            | beam.Map(lambda x: (None, x))
            | beam.ParDo(IndexAssigningStatefulDoFn())
            | beam.Map(print))


if __name__ == "__main__":
    run()
