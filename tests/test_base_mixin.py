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

import pytest

import sawatabi.constants as constants
from sawatabi.base_mixin import BaseMixin

################################
# Base Mixin
################################


def test_base_mixin_article():
    base = BaseMixin()

    article = base._get_article("apple")
    assert article == "an"

    article = base._get_article(name="banana")
    assert article == "a"


def test_modeltype_to_vartype():
    base = BaseMixin()

    vartype = base._modeltype_to_vartype(constants.MODEL_ISING)
    assert vartype == "SPIN"

    vartype = base._modeltype_to_vartype(mtype=constants.MODEL_QUBO)
    assert vartype == "BINARY"

    with pytest.raises(ValueError):
        base._modeltype_to_vartype(mtype="Another Type")


def test_vartype_to_modeltype():
    base = BaseMixin()

    mtype = base._vartype_to_modeltype("SPIN")
    assert mtype == constants.MODEL_ISING

    mtype = base._vartype_to_modeltype(vartype="BINARY")
    assert mtype == constants.MODEL_QUBO

    with pytest.raises(ValueError):
        base._vartype_to_modeltype(vartype="Another Type")


def test_check_argument():
    base = BaseMixin()

    testvar_str = "test variable"
    testvar_int = 12345

    base._check_argument_type(value=testvar_str, atype=str)
    with pytest.raises(TypeError):
        base._check_argument_type(testvar_str, int)
    base._check_argument_type(testvar_str, (str, int))
    base._check_argument_type(testvar_int, (str, int))

    testvar_tuple_str = ("test variable")
    testvar_tuple_empty = ()
    testvar_tuple_str_int = ("test variable", 12345)

    base._check_argument_type_in_tuple(values=testvar_tuple_str, atype=str)
    with pytest.raises(TypeError):
        base._check_argument_type_in_tuple(testvar_tuple_empty, str)
    with pytest.raises(TypeError):
        base._check_argument_type_in_tuple(testvar_tuple_str_int, str)
    base._check_argument_type_in_tuple(testvar_tuple_str_int, (str, int))

    testvar_list_str = ["test variable"]
    testvar_list_empty = []
    testvar_list_str_int = ["test variable", 12345]

    base._check_argument_type_in_list(values=testvar_list_str, atype=str)
    with pytest.raises(TypeError):
        base._check_argument_type_in_list(testvar_list_empty, str)
    with pytest.raises(TypeError):
        base._check_argument_type_in_list(testvar_list_str_int, str)
    base._check_argument_type_in_list(testvar_list_str_int, (str, int))
