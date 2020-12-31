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


class BaseMixin:
    def __init__(self):
        pass

    @staticmethod
    def _get_article(name):
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
                article = BaseMixin._get_article(typestr)
            raise TypeError(f"'{name}' must be {article} {typestr}.")

    @staticmethod
    def _check_argument_type_in_tuple(name, values, atype):
        if len(values) == 0:
            raise TypeError(f"'{name}' must not be an empty tuple.")
        if not isinstance(atype, tuple):
            atype = [atype]
        for v in values:
            is_ok = False
            for at in atype:
                if isinstance(v, at):
                    is_ok = True
            if not is_ok:
                atypestr = [at.__name__ for at in atype]
                raise TypeError(f"All elements in tuple '{name}' must be one of {atypestr}.")

    @staticmethod
    def _check_argument_type_in_list(name, values, atype):
        if len(values) == 0:
            raise TypeError(f"'{name}' must not be an empty list.")
        if not isinstance(atype, tuple):
            atype = [atype]
        for v in values:
            is_ok = False
            for at in atype:
                if isinstance(v, at):
                    is_ok = True
            if not is_ok:
                atypestr = [at.__name__ for at in atype]
                raise TypeError(f"All elements in list '{name}' must be one of {atypestr}.")

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
