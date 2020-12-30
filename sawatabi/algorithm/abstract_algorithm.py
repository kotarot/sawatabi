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
from apache_beam.transforms.userstate import CombiningValueStateSpec

from sawatabi.base_mixin import BaseMixin


class AbstractAlgorithm(BaseMixin):
    def __init__(self):
        pass

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
