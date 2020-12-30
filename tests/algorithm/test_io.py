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
import pytest

from sawatabi.algorithm.io import IO


def test_io_read_from_pubsub():
    with pytest.raises(TypeError):
        IO.read_from_pubsub()

    with pytest.raises(ValueError):
        IO.read_from_pubsub(project="", topic="")

    with pytest.raises(ValueError):
        IO.read_from_pubsub(project="test-project", topic="")

    with pytest.raises(ValueError):
        IO.read_from_pubsub(project="", topic="test-topic")

    fn = IO.read_from_pubsub(project="test-project", topic="test-topic")
    assert isinstance(fn, beam.transforms.ptransform._ChainedPTransform)
    assert fn.label == "ReadFromPubSub|Decode"

    fn = IO.read_from_pubsub(project="test-project", subscription="test-subscription")
    assert isinstance(fn, beam.transforms.ptransform._ChainedPTransform)
    assert fn.label == "ReadFromPubSub|Decode"


def test_io_read_from_pubsub_as_number():
    fn = IO.read_from_pubsub_as_number(project="test-project", topic="test-topic")
    assert isinstance(fn, beam.transforms.ptransform._ChainedPTransform)
    assert fn.label == "ReadFromPubSub|Decode|Filter|To int"


def test_io_read_from_pubsub_as_json():
    fn = IO.read_from_pubsub_as_json(project="test-project", topic="test-topic")
    assert isinstance(fn, beam.transforms.ptransform._ChainedPTransform)
    assert fn.label == "ReadFromPubSub|Decode|To JSON"


def test_io_read_from_text():
    with pytest.raises(TypeError):
        IO.read_from_text()

    with pytest.raises(OSError):
        IO.read_from_text(path="")

    with pytest.raises(OSError):
        IO.read_from_text(path="/path/to/test/file")

    fn = IO.read_from_text(path="tests/algorithm/numbers_100.txt")
    assert isinstance(fn, beam.io.textio.ReadFromText)
    assert fn.label == "ReadFromText"


def test_io_read_from_text_as_number():
    fn = IO.read_from_text_as_number(path="tests/algorithm/numbers_100.txt")
    assert isinstance(fn, beam.transforms.ptransform._ChainedPTransform)
    assert fn.label == "ReadFromText|Filter|To int"


def test_io_read_from_text_as_json():
    fn = IO.read_from_text_as_json(path="tests/algorithm/numbers_100.json")
    assert isinstance(fn, beam.transforms.ptransform._ChainedPTransform)
    assert fn.label == "ReadFromText|To JSON"


def test_io_write_to_stdout():
    fn = IO.write_to_stdout()
    assert isinstance(fn, beam.transforms.ptransform._NamedPTransform)
    assert fn.label == "Print to stdout"


def test_io_write_to_pubsub():
    with pytest.raises(TypeError):
        IO.write_to_pubsub()

    with pytest.raises(ValueError):
        IO.write_to_pubsub(project="", topic="")

    with pytest.raises(ValueError):
        IO.write_to_pubsub(project="test-project", topic="")

    with pytest.raises(ValueError):
        IO.write_to_pubsub(project="", topic="test-topic")

    fn = IO.write_to_pubsub(project="test-project", topic="test-topic")
    assert isinstance(fn, beam.transforms.ptransform._ChainedPTransform)
    assert fn.label == "Encode|WriteToPubSub"


def test_io_write_to_text():
    fn = IO.write_to_text(path="/path/to/output")
    assert isinstance(fn, beam.io.textio.WriteToText)
    assert fn.label == "WriteToText"
