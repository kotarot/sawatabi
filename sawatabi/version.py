#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from codecs import open
from os import path
import re

def version():
    this_dir = path.abspath(path.dirname(__file__))
    with open(path.join(this_dir, '__init__.py'), encoding='utf8') as f:
        version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', f.read()).group(1)
        return version
