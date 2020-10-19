# Copyright 2020 Kotaro Terada, Shingo Furuyama, Junya Usui, and Kazuki Ono
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
import neal

import sawatabi.constants as constants
from sawatabi.model.physical_model import PhysicalModel
from sawatabi.solver.abstract_solver import AbstractSolver


class LocalSolver(AbstractSolver):
    def __init__(self, exact=False):
        self._exact = exact

    def solve(self, model):
        self._check_argument_type("model", model, PhysicalModel)

        linear, quadratic = {}, {}
        for k, v in model._interactions[constants.INTERACTION_BODY_LINEAR].items():
            linear[k] = -1.0 * v
        for k, v in model._interactions[constants.INTERACTION_BODY_QUADRATIC].items():
            quadratic[k] = -1.0 * v

        if model.get_type() == constants.MODEL_ISING:
            vartype = dimod.SPIN
        elif model.get_type() == constants.MODEL_QUBO:
            vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, model._offset, vartype)

        if self._exact:
            # dimod's brute force solver
            sampleset = dimod.ExactSolver().sample(bqm)

        else:
            # Simulated annealing (SA)
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm)

        return sampleset
