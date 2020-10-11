#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import re

from sawatabi import version

def test_version_format():
    assert re.match(r'^\d+.\d+.\d+(.(dev|a|b|rc)\d+)?$', version())
