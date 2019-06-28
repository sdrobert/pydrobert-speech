# Copyright 2019 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classes for post-processing feature matrices"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import warnings

from itertools import count

import numpy as np

from pydrobert.speech import AliasedFactory
from pydrobert.speech.util import read_signal

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

__all__ = [
    'PostProcessor',
    'Standardize',
    'CMVN',
    'Deltas',
]


class PostProcessor(AliasedFactory):
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


class Standardize(PostProcessor):
    r'''Standardize each feature coefficient

    Though the exact behaviour of an instance varies according to below,
    the "goal" of this transformation is such that every feature
    coefficient on the chosen axis has mean 0 and variance 1
    (if `norm_var` is ``True``) over the other axes. Features are
    assumed to be real; the return data type after ``apply()`` is always
    a 64-bit float.

    If `rfilename` is not specified or the associated file is empty,
    coefficients are standardized locally (within the target tensor). If
    `rfilename` is specified, feature coefficients are standardized
    according to the sufficient statistics collected in the file. The
    latter implementation is based off [povey2011]_. The statistics will be
    loaded with ``read_signal()``.

    Parameters
    ----------
    rfilename : str, optional
    norm_var : bool

    Additional keyword arguments can be passed to the initializer if
    rfilename is set. They will be passed on to `read_signal`

    Attributes
    ----------
    have_stats : bool

    See Also
    --------
    pydrobert.speech.util.read_signal
        Describes the strategy used for loading signals
    '''

    aliases = {'standardize', 'normalize', 'unit', 'cmvn'}

    def __init__(self, rfilename=None, norm_var=True, **kwargs):
        self._stats = None
        self._norm_var = bool(norm_var)
        if rfilename is not None:
            if 'dtype' in kwargs:
                self._stats = read_signal(rfilename, **kwargs)
            else:
                for dtype in (np.float64, np.float32, 'dm', 'fm'):
                    try:
                        self._stats = read_signal(
                            rfilename, dtype=dtype, **kwargs)
                        break
                    except (IOError, ValueError, ImportError, TypeError):
                        pass
                if self._stats is None:
                    raise IOError(
                        'Unable to load stats from {}'.format(rfilename))
                if len(self._stats.shape) == 1:
                    # stats were likely stored as simple binary. Need
                    # to make sure we've cast to the right kind of
                    # float. Probably a non-issue if we saved the data
                    # ourselves
                    self._sanitize_stats()
        elif kwargs:
            raise TypeError(
                'Invalid keyword arguments: {}'.format(tuple(kwargs)))
        super(Standardize, self).__init__()

    def _sanitize_stats(self, checked_other_float=False):
        try:
            self._stats = self._stats.reshape((2, -1))
            valid = np.isclose(
                np.round(self._stats[0, -1]), self._stats[0, -1])
            valid &= np.all(self._stats >= 0)
        except ValueError:
            # in this case we couldn't reshape to (2, -1).
            valid = False
        if not valid and checked_other_float:
            raise IOError(
                'Could not properly load statistics. Try specifying '
                'additional parameters in init (see docstring)')
        elif not valid:
            if self._stats.dtype not in (np.float32, np.float64):
                raise ValueError(
                    'Statistics were loaded with a weird data type ({}) and '
                    'are invalid. Make sure the arguments you passed to '
                    'the init are correct'.format(self._stats.dtype))
            elif self._stats.dtype == np.float32:
                self._stats = np.frombuffer(
                    self._stats.tobytes(), dtype=np.float64)
            else:
                self._stats = np.frombuffer(
                    self._stats.tobytes(), dtype=np.float32).astype(np.float64)
            self._sanitize_stats(True)

    @property
    def have_stats(self):
        '''Whether at least one feature vector has been accumulated'''
        return self._stats is not None and self._stats[0, -1]

    def _accumulate_vector(self, vec):
        # accumulate over a single feature vector
        num_coeffs = len(vec)
        if self._stats is None:
            self._stats = np.zeros((2, num_coeffs + 1), dtype=np.float64)
        elif self._stats.shape[1] != num_coeffs + 1:
            raise ValueError(
                'Expected feature vector of length {}; got {}'.format(
                    self._stats.shape[1] - 1, num_coeffs)
            )
        self._stats[0, -1] += 1
        self._stats[0, :-1] += vec.astype(np.float64)
        self._stats[1, :-1] += np.square(vec, dtype=np.float64)

    def _accumulate_tensor(self, tensor, axis):
        # accumulate over a tensor (with a shape)
        num_coeffs = tensor.shape[axis]
        if self._stats is None:
            self._stats = np.zeros((2, num_coeffs + 1), dtype=np.float64)
        elif self._stats.shape[1] != num_coeffs + 1:
            raise ValueError(
                'Expected feature vector of length {}; got {}'.format(
                    self._stats.shape[1] - 1, num_coeffs)
            )
        other_axes = tuple(
            idx for idx in range(len(tensor.shape))
            if idx != axis % len(tensor.shape)
        )
        self._stats[0, -1] += np.prod(tuple(
            tensor.shape[idx] for idx in other_axes
        ))
        self._stats[0, :-1] += tensor.sum(axis=other_axes, dtype=np.float64)
        self._stats[1, :-1] += np.square(
            tensor, dtype=np.float64).sum(axis=other_axes)

    def accumulate(self, features, axis=-1):
        '''Accumulate statistics from a feature tensor

        Parameters
        ----------
        features : array-like
        axis : int, optional

        Raises
        ------
        IndexError
        ValueError
            If the length of `axis` does not match that of past feature
            vector lengths
        '''
        if (features.shape and not np.prod(features.shape)) or \
                not len(features):
            raise ValueError('Cannot accumulate from empty array')
        if features.shape and len(features.shape) > 1:
            self._accumulate_tensor(features, axis)
        else:
            self._accumulate_vector(features)

    def _apply_vector(self, vec, in_place):
        # apply transformation to vector
        num_coeffs = len(vec)
        if self._stats is not None and self._stats.shape[1] != num_coeffs + 1:
            raise ValueError(
                'Expected feature vector of length {}; got {}'.format(
                    self._stats.shape[1] - 1, num_coeffs)
            )
        if not in_place or vec.dtype != np.float64:
            vec = vec.astype(np.float64)
        if self.have_stats:
            count = self._stats[0, -1]
            means = self._stats[0, :-1] / count
            if self._norm_var:
                varss = self._stats[1, :-1] / count - means ** 2
                close_zero = np.isclose(varss, 0)
                if np.any(close_zero):
                    warnings.warn('0 variance encountered. Replacing with 1')
                    varss[close_zero] = 1
                scales = 1 / (varss ** .5)
            else:
                scales = 1
            vec *= scales
            vec -= means * scales
        else:
            if self._norm_var:
                raise ValueError(
                    'Unable to standardize the variance of a vector '
                    'with no global statistics'
                )
            else:
                warnings.warn('Standardizing a single vector to 0')
                vec[...] = 0
        return vec

    def _apply_tensor(self, tensor, axis, in_place):
        # apply transformation to tensor (with shape)
        num_coeffs = tensor.shape[axis]
        if self._stats is not None and self._stats.shape[1] != num_coeffs + 1:
            raise ValueError(
                'Expected feature vector of length {}; got {}'.format(
                    self._stats.shape[1] - 1, num_coeffs)
            )
        other_axes = tuple(
            idx for idx in range(len(tensor.shape))
            if idx != axis % len(tensor.shape)
        )
        if not in_place or tensor.dtype != np.float64:
            tensor = tensor.astype(np.float64)
        if self.have_stats:
            count = self._stats[0, -1]
            means = self._stats[0, :-1] / count
            varss = self._stats[1, :-1] / count - means ** 2
        elif sum(tensor.shape[idx] for idx in other_axes) == len(other_axes):
            if self._norm_var:
                raise ValueError(
                    'Unable to standardize the variance of a vector '
                    'with no global statistics'
                )
            else:
                warnings.warn('Standardizing a single vector to 0')
                tensor[...] = 0
                return tensor
        else:
            count = np.prod(tuple(tensor.shape[idx] for idx in other_axes))
            means = tensor.mean(axis=other_axes)
            varss = (tensor ** 2).sum(axis=other_axes) / count - means ** 2
        if self._norm_var:
            close_zero = np.isclose(varss, 0)
            if np.any(close_zero):
                warnings.warn('0 variance encountered. Replacing with 1')
                varss[close_zero] = 1
            scales = 1 / (varss ** .5)
        else:
            scales = np.ones(1)
        tensor_slice = [None] * len(tensor.shape)
        tensor_slice[axis] = slice(None)
        tensor_slice = tuple(tensor_slice)
        tensor *= scales[tensor_slice]
        tensor -= (means * scales)[tensor_slice]
        return tensor

    def apply(self, features, axis=-1, in_place=False):
        if (features.shape and not np.prod(features.shape)) or \
                not len(features):
            raise ValueError('Cannot apply to empty array')
        if features.shape and len(features.shape) > 1:
            return self._apply_tensor(features, axis, in_place)
        else:
            return self._apply_vector(features, in_place)

    def save(self, wfilename, key=None, compress=False, overwrite=True):
        r'''Save accumulated statistics to file

        If `wfilename` ends in `.npy`, stats will be written using
        `np.save`.

        If `wfilename` ends in `.npz`, stats will be written to a numpy
        archive. If `overwrite` is `False`, other key-values will be
        loaded first if possible, then resaved. If `key` is set, data
        will be indexed by `key` in the archive. Otherwise, the data
        will be stored at the first unused key of the pattern
        ``'arr_\d+'``. If `compress` is `True`,
        `numpy.savez_compressed` will be used over `numpy.savez`.

        Otherwise, data will be written using `np.ndarray.tofile`

        Parameters
        ----------
        wfilename : str
        key : str, optional
        compress : bool
        overwrite : bool

        Raises
        ------
        ValueError
            If no stats have been accumulated
        '''
        if not self.have_stats:
            raise ValueError('No stats have been accumulated to save')
        if wfilename.endswith('.npy'):
            np.save(wfilename, self._stats)
        elif wfilename.endswith('.npz'):
            array = dict()
            if overwrite:
                try:
                    array = np.load(wfilename)
                except IOError:
                    pass
            if key is None:
                for key in ('arr_{}'.format(v) for v in count(0)):
                    if key not in array:
                        break
            array[key] = self._stats
            if compress:
                np.savez_compressed(wfilename, **array)
            else:
                np.savez(wfilename, **array)
        else:
            self._stats.tofile(wfilename)


CMVN = Standardize


class Deltas(PostProcessor):
    r'''Calculate feature deltas (weighted rolling averages)

    Deltas are calculated by correlating the feature tensor with a 1D
    delta filter by enumerating over all but one axis (the "time axis"
    equivalent).

    Intermediate values are calculated with 64-bit floats, then cast
    back to the input data type.

    `Deltas` will increase the size of the feature tensor when
    `num_deltas` is positive and passed features are non-empty.

    If `concatenate` is ``True``, `target_axis` specifies the axis along
    which new deltas are appended. For example,

    >>> deltas = Deltas(num_deltas=2, concatenate=True, target_axis=1)
    >>> features_shape = list(features.shape)
    >>> features_shape[1] *= 3
    >>> assert deltas.apply(features).shape == tuple(features_shape)

    If `concatenate` is ``False``, `target_axis` dictates the location
    of a new axis in the resulting feature tensor that will index the
    deltas (0 for the original features, 1 for deltas, 2 for double
    deltas, etc.). For example:

    >>> deltas = Deltas(num_deltas=2, concatenate=False, target_axis=1)
    >>> features_shape = list(features.shape)
    >>> features_shape.insert(1, 3)
    >>> assert deltas.apply(features).shape == tuple(features_shape)

    Deltas act as simple low-pass filters. Flipping the direction of the
    real filter to turn the delta operation into a simple convolution,
    the first order delta is defined as

    .. math::

         f(t) = \begin{cases}
            \frac{-t}{Z} & -W \leq t \leq W \\
            0 & \mathrm{else}
         \end{cases}

    where

    .. math:: Z = \sqrt{\frac{2W^3}{3}}

    For some :math:`W`. Its Fourier transform is

    .. math::

         F(\omega) = \frac{-2i}{Z\omega^2}\left(
            W\omega \cos W\omega - \sin W\omega \right)

    Note that it is completely imaginary. For :math:`W \geq 2`, :math:`F` is
    bound below :math:`\frac{i}{\omega}`. Hence, :math:`F` exhibits low-pass
    characteristics. Second order deltas are generating by convolving
    :math:`f(-t)`` with itself, third order is an additional :math:`f(-t)``,
    etc. By the convolution theorem, higher order deltas have Fourier responses
    that become tighter around :math:`F(0)`` (more lowpass).

    Parameters
    ----------
    num_deltas : int
    target_axis : int, optional
    concatenate : bool
    context_window : int, optional
        The length of the filter to either side of the window. Positive
    pad_mode : str or function, optional
        How to pad the input sequence when correlating. Additional
        keyword arguments will be passed to ``numpy.pad()``

    Attributes
    ----------
    num_deltas : int
    concatenate : bool
    '''

    aliases = {'deltas'}

    def __init__(
            self, num_deltas, target_axis=-1, concatenate=True,
            context_window=2, pad_mode='edge', **kwargs):
        self._target_axis = target_axis
        self._pad_mode = pad_mode
        self._pad_kwargs = kwargs
        self.concatenate = bool(concatenate)
        self.num_deltas = num_deltas
        self._filts = [np.ones(1, dtype=np.float64)]
        delta_filter = np.arange(1 + 2 * context_window, dtype=np.float64)
        delta_filter -= context_window
        delta_filter /= np.sum(delta_filter ** 2)
        for idx in range(num_deltas):
            self._filts.append(np.convolve(self._filts[idx], delta_filter))

    def apply(self, features, axis=-1, in_place=False):
        delta_feats = [features]
        other_axes = tuple(
            idx for idx in range(len(features.shape))
            if idx != axis % len(features.shape)
        )
        other_shapes = tuple(features.shape[idx] for idx in other_axes)
        feat_slice = [slice(None)] * len(features.shape)
        for filt in self._filts[1:]:
            delta_feat = np.empty(features.shape, dtype=features.dtype)
            max_offset = (len(filt) - 1) // 2
            for other_indices in np.ndindex(other_shapes):
                for axis_idx, idx in zip(other_axes, other_indices):
                    feat_slice[axis_idx] = idx
                delta_feat[tuple(feat_slice)] = np.correlate(
                    np.pad(
                        features[tuple(feat_slice)].astype(
                            np.float64, copy=False),
                        (max_offset, max_offset),
                        self._pad_mode,
                        **self._pad_kwargs
                    ),
                    filt,
                    'full'
                )[len(filt) - 1:-len(filt) + 1].astype(
                    features.dtype, copy=False)
            delta_feats.append(delta_feat)
        if self.concatenate:
            return np.concatenate(delta_feats, self._target_axis)
        else:
            return np.stack(delta_feats, self._target_axis)
