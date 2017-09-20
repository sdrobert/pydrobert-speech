"""Signal processing library, primarily for speech

Available modules
-----------------
compute
    Feature computations/transformations from signals
config
    Package constants
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

__all__ = [
    'compute',
    'config',
    'filters',
    'post',
    'scales',
    'util',
]
