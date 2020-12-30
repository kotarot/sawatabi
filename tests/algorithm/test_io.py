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

import pytest

import apache_beam as beam

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
