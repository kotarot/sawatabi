#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import re

from sawatabi import version, version_info

def test_version_format():
    assert re.match(r'^\d+.\d+.\d+(.(dev|a|b|rc)\d+)?$', version())

def test_version_info_format():
    ver_info = version_info()
    assert (len(ver_info) == 3) or (len(ver_info) == 4)

    assert isinstance(ver_info[0], int)
    assert isinstance(ver_info[1], int)
    assert isinstance(ver_info[2], int)
    if len(ver_info) == 4:
        assert isinstance(ver_info[3], str)
