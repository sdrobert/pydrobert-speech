from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from tempfile import NamedTemporaryFile

import pytest

# throws away the user error on import if LC_ALL=C is not set
try:
    import pydrobert.kaldi
except ImportError:
    pass

warnings.simplefilter('error')
# annoying scipy errors. Not mah fault!
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", category=ImportWarning)


# fixtures
@pytest.fixture
def temp_file_1_name():
    temp = NamedTemporaryFile(suffix='_1', delete=False)
    temp.close()
    yield temp.name
    os.remove(temp.name)


@pytest.fixture
def temp_file_2_name():
    temp = NamedTemporaryFile(suffix='_2', delete=False)
    temp.close()
    yield temp.name
    os.remove(temp.name)
