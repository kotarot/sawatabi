#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from codecs import open
from os import path
import re
from setuptools import setup

package_name = 'sawatabi'
packages = [
    package_name,
    package_name + '.utils',
]

root_dir = path.abspath(path.dirname(__file__))

with open(path.join(root_dir, package_name, '__version__.py'), encoding='utf8') as f:
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', f.read()).group(1)

with open(path.join(root_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    packages=packages,
    version=version,
    license='Apache 2.0',
    install_requires=[
        'pyqubo>=0.4.0,<1.0.0',
        'dwave-neal>=0.5.6,<1.0.0',
        'apache-beam>=2.24.0,<3.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.1.1,<7.0.0',
            'pytest-cov>=2.10.1,<3.0.0',
            'black>=20.8b1,<21.0',
            'flake8>=3.8.4,<4.0.0',
            'isort>=5.6.4,<6.0.0',
        ],
    },
    author='Kotaro Terada, Shingo Furuyama, Junya Usui, and Kazuki Ono',
    author_email='kotarot@apache.org',
    url='https://github.com/kotarot/sawatabi',
    description='sawatabi.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='',
    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
    ],
)
