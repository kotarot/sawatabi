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

from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

import sawatabi.constants as constants
from sawatabi.model.physical_model import PhysicalModel
from sawatabi.solver.abstract_solver import AbstractSolver


class DWaveSolver(AbstractSolver):
    def __init__(self, endpoint=None, token=None, solver="Advantage_system1.1", embedding_parameters=None):
        super().__init__()
        self._endpoint = endpoint
        self._token = token
        self._solver = solver

        self._composite = self._create_composite(embedding_parameters)

    def _create_composite(self, embedding_parameters=None):
        if (self._endpoint is not None) and (self._token is not None):
            sampler = DWaveSampler(endpoint=self._endpoint, token=self._token, solver=self._solver)
        else:
            sampler = DWaveSampler(solver=self._solver)

        if (embedding_parameters is not None) and isinstance(embedding_parameters, dict):
            solver = EmbeddingComposite(sampler, embedding_parameters=embedding_parameters)
        else:
            solver = EmbeddingComposite(sampler)

        return solver

    def solve(self, model, **kwargs):
        self._check_argument_type("model", model, PhysicalModel)

        if len(model._raw_interactions[constants.INTERACTION_LINEAR]) == 0 and len(model._raw_interactions[constants.INTERACTION_QUADRATIC]) == 0:
            raise ValueError("Model cannot be empty.")

        # Converts to BQM (model representation for D-Wave)
        bqm = model.to_bqm()

        sampleset = self._composite.sample(bqm, **kwargs)

        return sampleset
