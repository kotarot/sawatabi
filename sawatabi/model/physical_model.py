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

import pprint

import dimod

import sawatabi.constants as constants
from sawatabi.model.abstract_model import AbstractModel


class PhysicalModel(AbstractModel):
    def __init__(self, mtype=""):
        super().__init__(mtype)
        self._raw_interactions = {
            constants.INTERACTION_LINEAR: {},  # linear (1-body)
            constants.INTERACTION_QUADRATIC: {},  # quadratic (2-body)
        }
        self._offset = 0.0
        self._variables_set = set()
        self._label_to_index = {}
        self._index_to_label = {}

    ################################
    # Interaction
    ################################

    def add_interaction(self, name, body, coefficient):
        self._raw_interactions[body][name] = coefficient

    ################################
    # Offset
    ################################

    def get_offset(self):
        """
        Returns the offset value.
        """
        return self._offset

    ################################
    # Converts to another model
    ################################

    def to_bqm(self, sign=-1.0):
        # Signs for BQM are opposite from our (sawatabi's) definition.
        # - BQM:      H =   sum( J_{ij} * x_i * x_j ) + sum( h_{i} * x_i )
        # - Sawatabi: H = - sum( J_{ij} * x_i * x_j ) - sum( h_{i} * x_i )
        linear, quadratic = {}, {}
        for k, v in self._raw_interactions[constants.INTERACTION_LINEAR].items():
            linear[k] = sign * v
        for k, v in self._raw_interactions[constants.INTERACTION_QUADRATIC].items():
            quadratic[k] = sign * v

        if self.get_mtype() == constants.MODEL_ISING:
            vartype = dimod.SPIN
        elif self.get_mtype() == constants.MODEL_QUBO:
            vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, self._offset, vartype)

        return bqm

    def to_polynomial(self):
        # For optigan, a variable identifier must be an integer.
        # Names for variables in the physical model is string, we need to convert them.
        #
        # Signs for Optigan are opposite from our (sawatabi's) definition.
        # - Optigan:  H =   sum( Q_{ij} * x_i * x_j ) + sum( Q_{i, i} * x_i )
        # - Sawatabi: H = - sum( J_{ij} * x_i * x_j ) - sum( h_{i} * x_i )
        polynomial = []
        for k, v in self._raw_interactions[constants.INTERACTION_LINEAR].items():
            index = self._label_to_index[k]
            polynomial.append([index, index, -1.0 * v])
        for k, v in self._raw_interactions[constants.INTERACTION_QUADRATIC].items():
            index = [self._label_to_index[k[0]], self._label_to_index[k[1]]]
            polynomial.append([index[0], index[1], -1.0 * v])

        return polynomial

    ################################
    # Built-in functions
    ################################

    def __eq__(self, other):
        return (
            isinstance(other, PhysicalModel)
            and (self._mtype == other._mtype)
            and (self._raw_interactions == other._raw_interactions)
            and (self._offset == other._offset)
            and (self._label_to_index == other._label_to_index)
            and (self._index_to_label == other._index_to_label)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        s = "PhysicalModel({"
        s += "'mtype': '" + str(self._mtype) + "', "
        s += "'raw_interactions': " + str(self._raw_interactions) + "}), "
        s += "'offset': " + str(self._offset)
        return s

    def __str__(self):
        s = []
        s.append("┏" + ("━" * 64))
        s.append("┃ PHYSICAL MODEL")
        s.append("┣" + ("━" * 64))
        s.append("┣━ mtype: " + str(self._mtype))
        s.append("┣━ raw_interactions:")
        s.append("┃  linear:")
        s.append(self.append_prefix(pprint.pformat(self._raw_interactions[constants.INTERACTION_LINEAR]), length=4))
        s.append("┃  quadratic:")
        s.append(self.append_prefix(pprint.pformat(self._raw_interactions[constants.INTERACTION_QUADRATIC]), length=4))
        s.append("┣━ offset: " + str(self._offset))
        s.append("┗" + ("━" * 64))
        return "\n".join(s)
