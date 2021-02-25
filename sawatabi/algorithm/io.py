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

import json
import re

import apache_beam as beam


class IO:

    ################################
    # Input (Read)
    ################################

    @classmethod
    def _read_as_number(cls, messages):
        # fmt: off
        number_pattern = re.compile(r"^[0-9]+$")
        return (messages
            | "Filter" >> beam.Filter(number_pattern.match)
            | "To int" >> beam.Map(int))
        # fmt: on

    @classmethod
    def _read_as_json(cls, messages):
        # fmt: off
        return (messages
            | "To JSON" >> beam.Map(json.loads))
        # fmt: on

    @classmethod
    def read_from_pubsub(cls, project, topic=None, subscription=None):
        # fmt: off
        if topic is not None:
            messages = beam.io.ReadFromPubSub(topic=f"projects/{project}/topics/{topic}")
        elif subscription is not None:
            messages = beam.io.ReadFromPubSub(subscription=f"projects/{project}/subscriptions/{subscription}")
        return (messages
            | "Decode" >> beam.Map(lambda m: m.decode("utf-8")))
        # fmt: on

    @classmethod
    def read_from_pubsub_as_number(cls, project, topic=None, subscription=None):
        messages = cls.read_from_pubsub(project=project, topic=topic, subscription=subscription)
        return cls._read_as_number(messages)

    @classmethod
    def read_from_pubsub_as_json(cls, project, topic=None, subscription=None):
        messages = cls.read_from_pubsub(project=project, topic=topic, subscription=subscription)
        return cls._read_as_json(messages)

    @classmethod
    def read_from_text(cls, path):
        return beam.io.ReadFromText(file_pattern=path)

    @classmethod
    def read_from_text_as_number(cls, path):
        messages = cls.read_from_text(path)
        return cls._read_as_number(messages)

    @classmethod
    def read_from_text_as_json(cls, path):
        messages = cls.read_from_text(path)
        return cls._read_as_json(messages)

    ################################
    # Output (Write)
    ################################

    @classmethod
    def write_to_stdout(cls):
        return "Print to stdout" >> beam.Map(print)

    @classmethod
    def write_to_pubsub(cls, project, topic):
        # fmt: off
        return ("Encode" >> beam.Map(lambda s: s.encode("utf-8"))
            | beam.io.WriteToPubSub(topic=f"projects/{project}/topics/{topic}"))
        # fmt: on

    @classmethod
    def write_to_text(cls, path):
        return beam.io.WriteToText(file_path_prefix=path)
