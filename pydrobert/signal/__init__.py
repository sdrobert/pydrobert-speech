"""Signal processing library, primarily for speech

Available subpackages
---------------------
compute
    Feature computations/transformations from signals
filters
    Filters and filter banks
post
    Post-processing of features
scales
    Scaling functions, including psychoacoustic scales such as Bark or
    Mel scales
util
    Miscellaneous functions for signal processing
vis
    Visualization functions. Requires `matplotlib` be installed
"""

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

"""
The scipy implementation of the FFT can be much faster than the numpy
one. This is set automatically to ``True`` if `scipy.fftpack` can be
imported. It can be set to ``False`` to use the numpy implementation.
"""
USE_FFTPACK = False
try:
    from scipy import fftpack
    USE_FFTPACK = True
except ImportError:
    pass

"""
No function is compactly supported in both the time and Fourier domains,
but large regions of either domain can be very close to zero. This
value serves as a threshold for zero. The higher it is, the more
accurate computations will be, but the longer they will take
"""
EFFECTIVE_SUPPORT_THRESHOLD = 5e-4

"""Value used as floor when taking log in computations"""
LOG_FLOOR_VALUE = 1e-5
