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

import sawatabi.constants as constants


class BaseMixin:
    def __init__(self):
        pass

    @staticmethod
    def _get_artitle(name):
        article = ""
        if name[0].lower() in ["a", "e", "i", "o", "u"]:
            article = "an"
        else:
            article = "a"
        return article

    @staticmethod
    def _check_argument_type(name, value, atype):
        if not isinstance(value, atype):
            if isinstance(atype, tuple):
                typestr = [t.__name__ for t in atype]
                article = "one of"
            else:
                typestr = atype.__name__
                article = BaseMixin._get_artitle(typestr)
            raise TypeError("'{}' must be {} {}.".format(name, article, typestr))

    @staticmethod
    def _check_argument_type_in_tuple(name, values, atype):
        if len(values) == 0:
            raise TypeError("'{}' must not be an empty tuple.".format(name))
        for v in values:
            if not isinstance(v, atype):
                typestr = atype.__name__
                article = BaseMixin._get_artitle(typestr)
                raise TypeError("All elements in '{}' must be {} {}.".format(name, article, typestr))

    @staticmethod
    def _modeltype_to_vartype(mtype):
        if mtype == constants.MODEL_ISING:
            vartype = "SPIN"
        elif mtype == constants.MODEL_QUBO:
            vartype = "BINARY"
        else:
            raise ValueError("Invalid 'mtype'")
        return vartype

    @staticmethod
    def _vartype_to_modeltype(vartype):
        if vartype == "SPIN":
            mtype = constants.MODEL_ISING
        elif vartype == "BINARY":
            mtype = constants.MODEL_QUBO
        else:
            raise ValueError("Invalid 'vartype'")
        return mtype