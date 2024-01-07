import os
import warnings

from tempfile import NamedTemporaryFile, mkdtemp
from shutil import rmtree
from zlib import adler32


import pytest
import numpy as np

try:
    from pydrobert.kaldi import KaldiLocaleWarning  # type: ignore

    warnings.filterwarnings("ignore", category=KaldiLocaleWarning)
except ImportError:
    pass

warnings.simplefilter("error")
# # annoying scipy errors. Not mah fault!
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# fixtures
@pytest.fixture
def temp_file_1_name():
    temp = NamedTemporaryFile(suffix="_1", delete=False)
    temp.close()
    yield temp.name
    os.remove(temp.name)


@pytest.fixture
def temp_file_2_name():
    temp = NamedTemporaryFile(suffix="_2", delete=False)
    temp.close()
    yield temp.name
    os.remove(temp.name)


@pytest.fixture
def temp_dir():
    dir_name = mkdtemp()
    yield dir_name
    rmtree(dir_name)


def pytest_runtest_setup(item):
    # implicitly seeds all tests for the sake of reproducibility
    np.random.seed(abs(adler32(bytes(item.name, "utf-8"))))
