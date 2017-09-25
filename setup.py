# pylint: skip-file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from codecs import open
from os import path
from setuptools import setup

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version 2.7 or >= 3.4 required.")

PWD = path.abspath(path.dirname(__file__))
with open(path.join(PWD, 'README.rst'), encoding='utf-8') as readme_file:
    LONG_DESCRIPTION = readme_file.read()

setup(
    name='pydrobert-signal',
    description='Signal processing with Python',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/sdrobert/pydrobert-signal',
    author=__author__,
    author_email=__email__,
    license=__license__,
    namespace_packages=['pydrobert'],
    packages=['pydrobert', 'pydrobert.signal'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
        'numpy', 'six',
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest', 'scipy',
    ],
    extras_require={
        'vis': ['matplotlib'],
        'kaldi': ['pydrobert-kaldi'],
    },
    entry_points={
        'console_scripts': [
            'compute-feats-from-kaldi-tables = pydrobert.signal.command_line:c'
            'ompute_feats_from_kaldi_tables [kaldi]',
        ]
    }
)
