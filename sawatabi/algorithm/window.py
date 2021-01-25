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

import sawatabi
from sawatabi.algorithm.abstract_algorithm import AbstractAlgorithm


class Window(AbstractAlgorithm):
    @classmethod
    def create_pipeline(cls, algorithm_options, **kwargs):
        algorithm_transform = (
            "Sliding windows" >> beam.WindowInto(beam.window.SlidingWindows(size=algorithm_options["window.size"], period=algorithm_options["window.period"]))
            | "Add timestamp as tuple againt each window for diff detection" >> beam.ParDo(AbstractAlgorithm.WithTimestampTupleFn())
            | "Elements in a sliding window into a list" >> beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
            | "To a single global Window from sliding windows" >> beam.WindowInto(beam.window.GlobalWindows())
        )

        return cls._create_pipeline(
            algorithm=sawatabi.constants.ALGORITHM_WINDOW,
            algorithm_transform=algorithm_transform,
            algorithm_options=algorithm_options,
            **kwargs,
        )

    ################################
    # Built-in functions
    ################################

    def __repr__(self):
        return "Window()"
