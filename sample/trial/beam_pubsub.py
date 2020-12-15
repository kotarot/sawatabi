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
import logging

import apache_beam as beam
from apache_beam import coders
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.trigger import AccumulationMode, AfterAll, AfterAny, AfterCount, AfterProcessingTime, AfterWatermark, DefaultTrigger, Repeatedly
from apache_beam.transforms.userstate import BagStateSpec
from beam_trial import IndexAssigningStatefulDoFn, MyClass, MyLineLengthFn, MyStatefulDoFn, WithTimestampFn, WithTimestampTupleFn

"""
This script subscribes test messages from GCP Pub/Sub using Apache Beam.
Subscribed messages will be divided into fixed/sliding windows.
StatefulDoFn functions are also applied to the streaming data.

Sample Usage:
$ GOOGLE_APPLICATION_CREDENTIALS="sample/trial/gcp-key.json" python sample/trial/beam_pubsub.py --project=your-project --topic=your-topic
$ GOOGLE_APPLICATION_CREDENTIALS="sample/trial/gcp-key.json" python sample/trial/beam_pubsub.py --project=your-project --subscription=your-subsctiption
"""


# User-defined Stateful DoFn
class MyStatefulDoFn_ForStreaming(beam.DoFn):
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

        # generator into a list
        my_states = list(my.read())

        # Clear the BagState so we can hold only the latest state.
        my.clear()

        # If we have no state yet
        if len(my_states) == 0:
            m = MyClass(fvalue, -fvalue, max(value), min(value))
            m.update_dt()
            my.add(m)
        # Otherwise, update the existing state
        else:
            m = my_states[0]
            m.myadd(fvalue)
            m.mysub(fvalue)
            m.mymax(max(value))
            m.mymin(min(value))
            m.update_dt()
            my.add(m)

        #print(m)
        yield f"first={fvalue:2d}, last={lvalue:2d}, (acc)add={m.a:6d}, (acc)sub={m.s:6d}, (acc)max={m.x:2d}, (acc)min={m.y:2d}, dt={m.dt}, value={value}, len={len(value)}"


# Stateful DoFn for detecting window diffs
class WindowDiffStatefulDoFn_ForStreaming(beam.DoFn):
    PREV_STATE = BagStateSpec(name="prev", coder=coders.PickleCoder())

    def process(self, element, prev=beam.DoFn.StateParam(PREV_STATE)):
        _, value = element

        # Sort with the event time.
        # If we sort a list of tuples, the first element of the tuple is recognized as a key by default,
        # so just `sorted` is enough.
        sorted_value = sorted(value)

        # generator into a list
        prev_states = list(prev.read())

        # Clear the BagState so we can hold only the latest state.
        prev.clear()

        if len(prev_states) == 0:
            prev_value = []
        else:
            prev_value = prev_states[-1]
        prev.add(sorted_value)

        outgoing = []
        for p in prev_value:
            if p[0] >= sorted_value[0][0]:
                break
            outgoing.append(p)

        incoming = []
        if len(prev_value) == 0:
            incoming = sorted_value
        else:
            for v in reversed(sorted_value):
                if v[0] <= prev_value[-1][0]:
                    break
                incoming.insert(0, v)

        def extract(value_list):
            extracted = []
            for v in value_list:
                extracted.append(v[1])
            return extracted

        yield {"incoming": extract(incoming), "outgoing": extract(outgoing), "elements": extract(value), "sorted": extract(sorted_value), "len": len(value)}


def run(argv=None):
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
    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_args.extend([
        "--runner=DirectRunner",
    ])

    pipeline_options = PipelineOptions(pipeline_args, streaming=True, save_main_session=True)
    with beam.Pipeline(options=pipeline_options) as p:

        if known_args.topic:
            messages = (p
                | "Subscribe Pub/Sub messages" >> beam.io.ReadFromPubSub(topic=f"projects/{known_args.project}/topics/{known_args.topic}"))
        elif known_args.subscription:
            messages = (p
                | "Subscribe Pub/Sub messages" >> beam.io.ReadFromPubSub(subscription=f"projects/{known_args.project}/subscriptions/{known_args.subscription}"))

        decoded = (messages
            | "Decode" >> beam.Map(lambda x: x.decode("utf-8")))

        processed = (decoded
            | "ParDo MyLineLengthFn" >> beam.ParDo(MyLineLengthFn())
            | "Extract only value" >> beam.Values())

        fixed_windows = (processed
            | "Fixed window of 10 sec" >> beam.WindowInto(beam.window.FixedWindows(size=10))

            # Using triggers:
            #| "Fixed window of 10 sec with a 10-sec-delay processing time trigger" >> beam.WindowInto(
            #        beam.window.FixedWindows(size=10),
            #        trigger=AfterProcessingTime(delay=10),
            #        accumulation_mode=AccumulationMode.DISCARDING)
            #| "Fixed window of 10 sec with 10-sec-delay processing time trigger but must contain at least 10 elements" >> beam.WindowInto(
            #        beam.window.FixedWindows(size=10),
            #        trigger=AfterCount(count=10),
            #        accumulation_mode=AccumulationMode.ACCUMULATING)
            #| "Global window that has at least 10 elements" >> beam.WindowInto(
            #        beam.window.GlobalWindows(),
            #        trigger=Repeatedly(AfterCount(count=10)),
            #        accumulation_mode=AccumulationMode.DISCARDING)

            # To list, we have two ways to do that.
            # Reference: https://cloud.google.com/pubsub/docs/pubsub-dataflow#python
            | "Fixed Windows to list" >> beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
            #| "Add temp key" >> beam.Map(lambda element: (None, element))
            #| "Group by" >> beam.GroupByKey()
            #| "Abandon temp key" >> beam.MapTuple(lambda _, val: val)

            | "Global Window for fixed windows" >> beam.WindowInto(beam.window.GlobalWindows())
            | beam.Map(lambda x: (None, x))

            #| beam.ParDo(IndexAssigningStatefulDoFn())
            #| "With count visible" >> beam.Map(lambda val: (val[0], val[1], len(val[1])))
            | beam.ParDo(MyStatefulDoFn_ForStreaming())

            | "With timestamp for fixed windows" >> beam.ParDo(WithTimestampFn())
            #| beam.Map(print)
        )

        sliding_windows = (processed
            | "Sliding windows of 20 sec with 5 sec interval" >> beam.WindowInto(beam.window.SlidingWindows(size=20, period=5))
            | "Add timestamp tuple for diff detection" >> beam.ParDo(WithTimestampTupleFn())
            | "Sliding Windows to list" >> beam.CombineGlobally(beam.combiners.ToListCombineFn()).without_defaults()
            | "Global Window for sliding windows" >> beam.WindowInto(beam.window.GlobalWindows())
            | beam.Map(lambda x: (None, x))
            | beam.ParDo(WindowDiffStatefulDoFn_ForStreaming())

            | "With timestamp for sliding windows" >> beam.ParDo(WithTimestampFn())
            | beam.Map(print)
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
