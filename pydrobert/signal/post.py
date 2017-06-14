"""Classes for post-processing feature matrices"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np

from six import with_metaclass

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

class PostProcessor(object, with_metaclass(abc.ABCMeta)):
    '''A container for post-processing features with a transform'''

    @abc.abstractmethod
    def apply(self, features, axis=-1, in_place=False):
        '''Applies the transformation to a feature tensor

        Consult the class documentation for more details on what the
        transformation is.

        Parameters
        ----------
        features : array-like
        axis : int, optional
            The axis of `features` to apply the transformation along
        in_place : bool, optional
            Whether it is okay to modify features (``True``) or whether
            a copy should be made (``False``)

        Returns
        -------
        array-like
            The transformed features
        '''
        pass

