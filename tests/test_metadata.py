"""Test package metadata"""

import pydrobert.speech


def test_version():
    assert pydrobert.speech.__version__ != "inplace"
