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

import sawatabi.constants as constants
from sawatabi.base_mixin import BaseMixin


class AbstractModel(BaseMixin):
    def __init__(self, mtype=""):
        if mtype in [constants.MODEL_ISING, constants.MODEL_QUBO]:
            self._mtype = mtype
        else:
            raise ValueError("'mtype' must be one of {}.".format([constants.MODEL_ISING, constants.MODEL_QUBO]))

    def get_mtype(self):
        return self._mtype

    ################################
    # Built-in functions
    ################################

    def __repr__(self):
        raise NotImplementedError("#{self.class}##{__method__} must be implemented.")

    def __str__(self):
        raise NotImplementedError("#{self.class}##{__method__} must be implemented.")

    @staticmethod
    def remove_leading_spaces(lines):
        lines = lines.split("\n")
        return " ".join([ln.lstrip() for ln in lines])

    @staticmethod
    def append_prefix(lines, length):
        lines = lines.split("\n")
        return "\n".join(["┃" + (" " * length) + ln for ln in lines])
