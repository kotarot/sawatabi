#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import re
from codecs import open
from os import path

from setuptools import setup

package_name = "sawatabi"
packages = [
    package_name,
    package_name + ".algorithm",
    package_name + ".model",
    package_name + ".model.constraint",
    package_name + ".solver",
    package_name + ".utils",
]

root_dir = path.abspath(path.dirname(__file__))

with open(path.join(root_dir, package_name, "__version__.py"), encoding="utf8") as f:
    version = re.search(r"__version__\s*=\s*[\'\"](.+?)[\'\"]", f.read()).group(1)

with open(path.join(root_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=package_name,
    packages=packages,
    version=version,
    license="Apache 2.0",
    install_requires=[
        "apache-beam>=2.26.0,<3.0.0",
        "apache-beam[gcp]>=2.26.0,<3.0.0",
        "dwave-system>=1.2.1,<2.0.0",
        "dwave-neal>=0.5.6,<1.0.0",
        "pandas>=1.1.4,<2.0.0",
        "pyqubo>=0.4.0,<1.0.0",
        "PyYAML>=5.3.1,<6.0.0",
    ],
    extras_require={
        "dev": [
            # Beam interactive runner
            "apache-beam[interactive]>=2.26.0,<3.0.0",
            # For test
            "pytest>=6.1.2,<7.0.0",
            "pytest-cov>=2.10.1,<3.0.0",
            "pytest-mock>=3.3.1,<4.0.0",
            "pytest-timeout>=1.4.2,<2.0.0",
            # For lint
            "black>=20.8b1,<21.0",
            "flake8>=3.8.4,<4.0.0",
            "isort>=5.6.4,<6.0.0",
            "mypy>=0.800,<1.0",
            # For profiling visualization
            "snakeviz>=2.1.0,<3.0.0",
            # For samples
            "geopy>=2.0.0,<3.0.0",
        ],
    },
    author="Kotaro Terada",
    author_email="kotarot@apache.org",
    url="https://github.com/kotarot/sawatabi",
    description="Sawatabi is an application framework to develop and run stream-data-oriented Ising applications with quantum annealing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="python streaming stream-processing ising-model ising quantum-annealing annealing",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
