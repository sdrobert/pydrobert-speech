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
with open(path.join(PWD, 'README.md'), encoding='utf-8') as readme_file:
    LONG_DESCRIPTION = readme_file.read()

SETUP_REQUIRES = ['setuptools_scm']
if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
    SETUP_REQUIRES += ['pytest-runner']

# needed to call tests thru python setup.py test
# otherwise, multiprocessing hangs
if __name__ == "__main__":
    setup(
        name='pydrobert-speech',
        description='Speech processing with Python',
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        use_scm_version=True,
        zip_safe=False,
        url='https://github.com/sdrobert/pydrobert-speech',
        author=__author__,
        author_email=__email__,
        license=__license__,
        packages=['pydrobert', 'pydrobert.speech'],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        install_requires=[
            'numpy', 'future',
        ],
        setup_requires=SETUP_REQUIRES,
        tests_require=[
            'pytest', 'scipy',
        ],
        extras_require={
            'vis': ['matplotlib'],
            'kaldi': ['pydrobert-kaldi'],
            'pytorch': ['pydrobert-pytorch'],
        },
        entry_points={
            'console_scripts': [
                'compute-feats-from-kaldi-tables = pydrobert.speech.'
                'command_line:compute_feats_from_kaldi_tables [kaldi]',
                'signals-to-torch-feat-dir = pydrobert.speech.command_line:'
                'signals_to_torch_feat_dir [pytorch]',
            ]
        }
    )
