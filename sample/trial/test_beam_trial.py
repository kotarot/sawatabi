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
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from beam_trial import MyLineLengthFn


def test_pipeline():
    expected = [
        6,
        13,
        21,
    ]

    inputs = [
        "To be,",
        "or not to be,",
        "that is the question.",
    ]

    with TestPipeline() as p:
        actual = (p
            | beam.Create(inputs)
            | beam.ParDo(MyLineLengthFn())
            | beam.Values())

        assert_that(actual, equal_to(expected))


if __name__ == "__main__":
    test_pipeline()