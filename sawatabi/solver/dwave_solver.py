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

from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

import sawatabi.constants as constants
from sawatabi.model.physical_model import PhysicalModel
from sawatabi.solver.abstract_solver import AbstractSolver


class DWaveSolver(AbstractSolver):
    def __init__(self, solver="Advantage_system1.1"):
        super().__init__()
        self._solver = solver

    def solve(self, model, seed=None, chain_strength=2.0, annealing_time=20, num_reads=1000, answer_mode="histogram"):
        self._check_argument_type("model", model, PhysicalModel)

        if (
            len(model._interactions[constants.INTERACTION_LINEAR]) == 0
            and len(model._interactions[constants.INTERACTION_QUADRATIC]) == 0
        ):
            raise ValueError("Model cannot be empty.")

        # Converts to BQM (model representation for D-Wave)
        bqm = model.to_bqm()

        # TODO: Deal with reverse annealing.
        solver = EmbeddingComposite(DWaveSampler(solver=self._solver))
        sampleset = solver.sample(
            bqm, chain_strength=chain_strength, annealing_time=20, num_reads=num_reads, answer_mode=answer_mode
        )

        return sampleset
