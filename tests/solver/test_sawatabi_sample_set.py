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

from sawatabi.solver import SawatabiSampleSet

################################
# Sawatabi Sample Set
################################


def test_sawatabi_sample_set():
    sss = SawatabiSampleSet()
    sss.variables = ["a", "b", "c", "d"]

    sss.add_record([0, 0, 0, 0], -10.0)
    assert len(sss.record) == 1
    assert sss.record[0][0] == [0, 0, 0, 0]
    assert sss.record[0][1] == -10.0
    assert sss.record[0][2] == 1

    sss.add_record(record=[1, 1, 1, 1], energy=-20.0)
    assert len(sss.record) == 2
    assert sss.record[1][0] == [1, 1, 1, 1]
    assert sss.record[1][1] == -20.0
    assert sss.record[1][2] == 1

    samples = sss.samples()
    assert len(samples) == 2
    assert len(samples[0]) == 4
    assert samples[0]["a"] == 0
    assert samples[0]["b"] == 0
    assert samples[0]["c"] == 0
    assert samples[0]["d"] == 0
    assert len(samples[1]) == 4
    assert samples[1]["a"] == 1
    assert samples[1]["b"] == 1
    assert samples[1]["c"] == 1
    assert samples[1]["d"] == 1


################################
# Built-in functions
################################


def test_sawatabi_sample_set_repr():
    sss = SawatabiSampleSet()
    assert isinstance(sss.__repr__(), str)
    assert "SawatabiSampleSet" in sss.__repr__()
    assert "rows" in sss.__repr__()
    assert "samples" in sss.__repr__()
    assert "variables" in sss.__repr__()
