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
import datetime
import logging
import os
import random
import sys
import time

import apache_beam as beam
from apache_beam import coders
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.trigger import AccumulationMode, AfterCount, AfterProcessingTime, AfterWatermark
from apache_beam.transforms.userstate import BagStateSpec, CombiningValueStateSpec, StateSpec

random.seed(0)


# Standard DoFn
class MyLineLengthFn(beam.DoFn):
    # Note: Do we need __init__ ?
    def __init__(self):
        pass

    def process(self, element):
        yield (element, len(element))


# Reference: https://beam.apache.org/documentation/transforms/python/elementwise/withtimestamps/
class WithTimestampFn(beam.DoFn):
    def process(self, data, timestamp=beam.DoFn.TimestampParam):
        yield f"[{timestamp.to_utc_datetime()}] {data}"


class WithTimestampTupleFn(beam.DoFn):
    def process(self, data, timestamp=beam.DoFn.TimestampParam):
        #yield (int(timestamp), data)
        yield (float(timestamp), data)


# Sample CombineFn
# Reference: https://beam.apache.org/documentation/programming-guide/
class AverageFn(beam.CombineFn):
    def create_accumulator(self):
        return (0.0, 0)

    def add_input(self, _sum_count, _input):
        (_sum, count) = _sum_count
        return _sum + _input, count + 1

    def merge_accumulators(self, _accumulators):
        _sums, _counts = zip(*_accumulators)
        return sum(_sums), sum(_counts)

    def extract_output(self, _sum_count):
        (_sum, _count) = _sum_count
        return _sum / _count if _count else float("NaN")


class MyClass:
    def __init__(self, a, s, x, y):
        self.a = a
        self.s = s
        self.x = x
        self.y = y
        self.dt = datetime.datetime.now()

    def myadd(self, a):
        self.a += a

    def mysub(self, s):
        self.s -= s

    def mymax(self, x):
        self.x = max(x, self.x)

    def mymin(self, y):
        self.y = min(y, self.y)

    def update_dt(self):
        self.dt = datetime.datetime.now()

    def __repr__(self):
        return f"** MyClass ** {self.a} : {self.s} : {self.x} : {self.y} : {self.dt}"


# Simple Stateful DoFn
# Reference:
#   - https://beam.apache.org/blog/stateful-processing/
#   - https://github.com/apache/beam/blob/master/sdks/python/apache_beam/transforms/userstate_test.py
class IndexAssigningStatefulDoFn(beam.DoFn):
    # State: A single integer
    INDEX_STATE = CombiningValueStateSpec(name="index", coder=coders.PickleCoder(), combine_fn=sum)

    def process(self, element, index=beam.DoFn.StateParam(INDEX_STATE)):
        _, value = element
        current_index = index.read()
        index.add(1)
        yield (current_index, value)


# User-defined Stateful DoFn
class MyStatefulDoFn(beam.DoFn):
    # State: User-defined object
    MY_STATE = BagStateSpec(name="my", coder=coders.PickleCoder())
    #MY_STATE = BagStateSpec(name="my", coder=coders.FastPrimitivesCoder())

    def process(self, element, my=beam.DoFn.StateParam(MY_STATE)):
        # element[0] is a key
        # element[1] is a value, represents a window (as a list) 
        #print(element)
        _, value = element
        if isinstance(value, list):
            fvalue = value[0]
            lvalue = value[-1]

        #print(my)
        #print(my.read())
        #print(my.read().first)
        #print(my.read().second)

        # If we have no state yet
        if len(my.read().second) == 0:
            m = MyClass(fvalue, -fvalue, max(value), min(value))
            my.add(m)
        # Otherwise, update the existing state
        else:
            m = (my.read().second)[0]
            m.myadd(fvalue)
            m.mysub(fvalue)
            m.mymax(max(value))
            m.mymin(min(value))
        m.update_dt()

        #print(m)
        yield f"first={fvalue:2d}, last={lvalue:2d}, (acc)add={m.a:6d}, (acc)sub={m.s:6d}, (acc)max={m.x:2d}, (acc)min={m.y:2d}, dt={m.dt}, value={value}"


# Stateful DoFn for detecting window diffs
class WindowDiffStatefulDoFn(beam.DoFn):
    PREV_STATE = BagStateSpec(name="prev", coder=coders.PickleCoder())

    def process(self, element, prev=beam.DoFn.StateParam(PREV_STATE)):
        _, value = element

        if len(prev.read().second) == 0:
            prev_value = []
        else:
            prev_value = (prev.read().second)[-1]
        prev.add(value)

        outgoing = []
        for p in prev_value:
            if p[0] >= value[0][0]:
                break
            outgoing.append(p)

        incoming = []
        if len(prev_value) == 0:
            incoming = value
        else:
            for v in reversed(value):
                if v[0] <= prev_value[-1][0]:
                    break
                incoming.insert(0, v)

        def extract(value_list):
            extracted = []
            for v in value_list:
                extracted.append(v[1])
            return extracted

        yield {"incoming": extract(incoming), "outgoing": extract(outgoing), "value": extract(value)}


def run(argv=None):
    dirname = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        dest="input",
        default=f"{dirname}/input.txt",
        help="Input file to process.")
    parser.add_argument(
        "--output",
        dest="output",
        default=f"{dirname}/output.txt",
        help="Output file to write results to.")
    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_args.extend([
        "--runner=DirectRunner",
    ])

    pipeline_options = PipelineOptions(pipeline_args)
    with beam.Pipeline(options=pipeline_options) as p:

        lines = p | "Input" >> beam.io.ReadFromText(known_args.input)

        def add_timestamp(element):
            return beam.window.TimestampedValue(element, time.time())

        lines = lines | "Add timestamps to lines" >> beam.Map(add_timestamp)

        def my_print(text):
            print(f"my_print --> {text}")
            return text

        pre_processed = (lines
            #| "Print line" >> beam.Map(my_print)
            | "Filter empty line" >> beam.Filter(lambda text: len(text) != 0)
            #| "Print filtered line" >> beam.Map(my_print)
        )

        processed = pre_processed | "ParDo MyLineLengthFn" >> beam.ParDo(MyLineLengthFn())

        post_processed = (processed
            | "Swap key and value" >> beam.KvSwap()
            #| "Print count" >> beam.Map(my_print)
        )

        def format_result(length, text):
            return f"{length:2d} | {text}"

        formatted = post_processed | "Format" >> beam.MapTuple(format_result)
        output = formatted | "Output" >> beam.io.WriteToText(known_args.output)

        ################################################################

        processed_value = (processed
            | "Extract only value" >> beam.Values()
            #| "Print only value" >> beam.Map(my_print)
        )

        # Just want to calculate the average and the sum of each line length.
        #average = processed_value | beam.CombineGlobally(AverageFn())
        #average | "Print average" >> beam.Map(print)
        #sumval = processed_value | beam.CombineGlobally(sum).without_defaults()
        #sumval | "Print sum" >> beam.Map(print)

        ################################
        # Timestamp by processing time

        # Note: Data from text data source do not have timestamp by default.
        timestamp_processing_time = processed_value | "Add timestamps to length" >> beam.Map(add_timestamp)
        #_ = (timestamp_processing_time
        #    | "With timestamp" >> beam.ParDo(WithTimestampFn())
        #    | "Print timestamp" >> beam.Map(print))

        # Windowing by event time
        windows_by_event_time = (timestamp_processing_time
            | "Fixed window" >> beam.WindowInto(beam.window.FixedWindows(0.005)))

        # Print windows as a list
        #_ = (windows_by_event_time
        #    | "Windows by event time to list" >> beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
        #    | beam.Map(print))

        # Print averages for each window
        #_ = (windows_by_event_time
        #    | beam.CombineGlobally(AverageFn()).without_defaults()
        #    | "Windows by event time to list" >> beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
        #    | beam.Map(print))

        ################################
        # Timestamp by line number

        processed_kv = (processed_value
            | "Format Key-Value" >> beam.Map(lambda element: (None, element))
            #| beam.Map(print)
        )

        index_assigned = (processed_kv
            | "ParDo IndexAssigningStatefulDoFn" >> beam.ParDo(IndexAssigningStatefulDoFn())
            #| beam.Map(print)
        )

        # Add event timestamp based on the index
        def add_timestamp_based_on_index(event):
            # event[0]: index
            # event[1]: data
            #return beam.window.TimestampedValue(event[1], event[0])  # w/o variation
            return beam.window.TimestampedValue(event[1], event[0] + random.random() - 0.5)  # w/ random variation

        timestamp_by_index = index_assigned | "With timestamps by index" >> beam.Map(add_timestamp_based_on_index)
        #_ = (timestamp_by_index
        #    | "With timestamp" >> beam.ParDo(WithTimestampFn())
        #    | "Print timestamp" >> beam.Map(print))

        # Fixed Windowing by index
        fixed_windowing_by_index = (timestamp_by_index
            | "Fixed window 20" >> beam.WindowInto(beam.window.FixedWindows(size=20))
            | "Fixed Windows by index to list" >> beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
            | "Global Window for fixed windows" >> beam.WindowInto(beam.window.GlobalWindows())
            | beam.Map(lambda x: (None, x))
            | beam.ParDo(IndexAssigningStatefulDoFn())
            #| "Print fixed windows" >> beam.Map(print)
        )

        # Slinding Windowing by index
        sliding_windowing_by_index = (timestamp_by_index
            | "Sliding window 20" >> beam.WindowInto(beam.window.SlidingWindows(size=20, period=5))
            | "Sliding Windows by index to list" >> beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
            | "Global Window for sliding windows" >> beam.WindowInto(beam.window.GlobalWindows())
            | beam.Map(lambda x: (None, x))
            | beam.ParDo(MyStatefulDoFn())
            #| "Print sliding windows" >> beam.Map(print)
        )

        # Detect window diffs
        sliding_windowing_for_diff = (timestamp_by_index
            | "Sliding window for diff 20" >> beam.WindowInto(beam.window.SlidingWindows(size=20, period=5))
            | "Add timestamp tuple for diff detection" >> beam.ParDo(WithTimestampTupleFn())
            | "Sliding Windows for diff to list" >> beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
            | "Global Window for sliding windows for diff" >> beam.WindowInto(beam.window.GlobalWindows())
            | beam.Map(lambda x: (None, x))
            | beam.ParDo(WindowDiffStatefulDoFn())
            | "Print window diffs" >> beam.Map(print)
        )


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.INFO)
    run()
