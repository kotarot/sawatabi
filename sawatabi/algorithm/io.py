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

import json
import re

import apache_beam as beam

import sawatabi


class IO:

    ################################
    # Input (Read)
    ################################

    @staticmethod
    def _read_as_number(messages):
        # fmt: off
        number_pattern = re.compile(r"^[0-9]+$")
        return (messages
            | "Filter" >> beam.Filter(lambda element: number_pattern.match(element))
            | "To int" >> beam.Map(lambda e: int(e)))
        # fmt: on

    @staticmethod
    def _read_as_json(messages):
        # fmt: off
        return (messages
            | "To JSON" >> beam.Map(lambda e: json.loads(e)))
        # fmt: on

    @staticmethod
    def read_from_pubsub(project, topic=None, subscription=None):
        # fmt: off
        if topic is not None:
            messages = beam.io.ReadFromPubSub(topic=f"projects/{project}/topics/{topic}")
        elif subscription is not None:
            messages = beam.io.ReadFromPubSub(subscription=f"projects/{project}/subscriptions/{subscription}")
        return (messages
            | "Decode" >> beam.Map(lambda m: m.decode("utf-8")))
        # fmt: on

    @staticmethod
    def read_from_pubsub_as_number(project, topic=None, subscription=None):
        messages = sawatabi.algorithm.IO.read_from_pubsub(project=project, topic=topic, subscription=subscription)
        return sawatabi.algorithm.IO._read_as_number(messages)

    @staticmethod
    def read_from_pubsub_as_json(project, topic=None, subscription=None):
        messages = sawatabi.algorithm.IO.read_from_pubsub(project=project, topic=topic, subscription=subscription)
        return sawatabi.algorithm.IO._read_as_json(messages)

    @staticmethod
    def read_from_text(path):
        return beam.io.ReadFromText(file_pattern=path)

    @staticmethod
    def read_from_text_as_number(path):
        messages = sawatabi.algorithm.IO.read_from_text(path)
        return sawatabi.algorithm.IO._read_as_number(messages)

    @staticmethod
    def read_from_text_as_json(path):
        messages = sawatabi.algorithm.IO.read_from_text(path)
        return sawatabi.algorithm.IO._read_as_json(messages)

    ################################
    # Output (Write)
    ################################

    @staticmethod
    def write_to_stdout():
        return beam.Map(print)

    @staticmethod
    def write_to_pubsub(topic):
        return beam.io.WriteStringsToPubSub(topic=topic)

    @staticmethod
    def write_to_text(path):
        return beam.io.WriteToText(file_path_prefix=path)
