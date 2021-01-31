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

import time

import dimod
import neal

import sawatabi.constants as constants
from sawatabi.model.physical_model import PhysicalModel
from sawatabi.solver.abstract_solver import AbstractSolver


class LocalSolver(AbstractSolver):
    def __init__(self, exact=False):
        super().__init__()
        self._exact = exact

        if self._exact:
            # dimod's brute force solver
            self._solver = dimod.ExactSolver()
        else:
            # Simulated annealing (SA)
            self._solver = neal.SimulatedAnnealingSampler()

    def solve(self, model, **kwargs):
        self._check_argument_type("model", model, PhysicalModel)

        if len(model._raw_interactions[constants.INTERACTION_LINEAR]) == 0 and len(model._raw_interactions[constants.INTERACTION_QUADRATIC]) == 0:
            raise ValueError("Model cannot be empty.")

        bqm = model.to_bqm()

        start_sec = time.perf_counter()
        if self._exact:
            # dimod's brute force solver
            sampleset = self._solver.sample(bqm)
        else:
            # Simulated annealing (SA)
            sampleset = self._solver.sample(bqm, **kwargs)

        # Update the timing
        execution_sec = time.perf_counter() - start_sec
        sampleset.info["timing"] = {
            "execution_sec": execution_sec,
        }

        return sampleset

    def default_beta_range(self, model):
        self._check_argument_type("model", model, PhysicalModel)

        if len(model._raw_interactions[constants.INTERACTION_LINEAR]) == 0 and len(model._raw_interactions[constants.INTERACTION_QUADRATIC]) == 0:
            raise ValueError("Model cannot be empty.")

        return neal.default_beta_range(model.to_bqm())
