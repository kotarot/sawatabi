#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from codecs import open
from os import path
import re
from setuptools import setup

package_name = 'sawatabi'
root_dir = path.abspath(path.dirname(__file__))

with open(path.join(root_dir, package_name, '__init__.py'), encoding='utf8') as f:
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', f.read()).group(1)

with open(path.join(root_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    packages=[package_name],
    version=version,
    license='Apache 2.0',
    install_requires=[],
    extras_require={
        'dev': [
            'pytest>=6',
        ],
    },
    author='Kotaro Terada',
    author_email='kotarot@apache.org',
    url='https://github.com/kotarot/sawatabi',
    description='sawatabi.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='',
    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9',
    ],
)
