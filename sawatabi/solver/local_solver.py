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


class LocalSolver:
    def __init__(self):
        pass

    @staticmethod
    def _check_argument_type(name, value, type):
        if not isinstance(value, type):
            if isinstance(type, tuple):
                typestr = [t.__name__ for t in type]
                article = "one of"
            else:
                typestr = type.__name__
                if typestr[0] in ["a", "e", "i", "o", "u"]:
                    article = "an"
                else:
                    article = "a"
            raise TypeError("'{}' must be {} {}.".format(name, article, typestr))

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
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype)
        print("\n")
        print(bqm)

        # dimod's brute force solver
        # sampleset = dimod.ExactSolver().sample(bqm)
        # print("\n")
        # print(sampleset)

        # SA
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm)
        print("\n")
        print(sampleset)

        return sampleset
