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

import dimod


class SawatabiSampleSet:
    def __init__(self):
        self.info = None
        self.variables = []
        self.record = []
        self.vartype = dimod.BINARY
        self.first = None

    def add_record(self, record, energy):
        self.record.append((record, energy, 1))
        self.first = self.record[0]

    def samples(self):
        samples = []
        for r in self.record:
            samples.append(dict(zip(self.variables, r[0])))
        return samples

    def __repr__(self):
        return f"SawatabiSampleSet\n[{self.vartype}, {len(self.record)} rows, {len(self.record)} samples, {len(self.variables)} variables]"
