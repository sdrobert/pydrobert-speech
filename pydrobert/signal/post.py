"""Classes for post-processing feature matrices"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import warnings

from os.path import exists
from re import match

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

class Standardize(PostProcessor):
    r'''Standardize each feature coefficient

    Though the exact behaviour of an instance varies according to below,
    the "goal" of this transformation is such that every feature
    coefficient on the chosen axis has mean 0 and variance 1
    (if `norm_var` is ``True``) over the other axes.

    If `rfilename` is not specified or the associated file is empty,
    coefficients are standardized locally (within the target tensor). If
    `rfilename` is specified, feature coefficients are standardized
    according to the sufficient statistics collected in the file. The
    latter implementation is based off [1]_.

    If `rfilename` begins with the regex ``r'^(ark|scp)(,\w+)*:'``, the
    file is assumed to be a Kaldi archive or script. `Standardize` will
    attempt to import `pydrobert.kaldi.tables` in order to open it
    (see pydrobert-kaldi_ for more details). Otherwise, it will be
    saved/loaded in Numpy (``.npy``) format.

    Parameters
    ----------
    rfilename : str, optional
    key : str, optional
        Different stats can be stored/retrieved from the same file using
        key/value pairs. If `key` is set, `rfilename` is assumed to store
        key/value pairs, and the stats associated with `key` are loaded.
    norm_var : bool, optional

    Attributes
    ----------
    have_stats : bool

    Raises
    ------
    ImportError
    KeyError

    References
    ----------
    .. _pydrobert-kaldi: https://github.com/sdrobert/pydrobert-kaldi
    .. [1] Povey, D., et al (2011). The Kaldi Speech Recognition
           Toolkit. ASRU
    '''

    BOGUS_KEY = 'stats'
    '''Key used when stats are written to a kaldi table without a key'''

    def __init__(self, rfilename=None, key=None, norm_var=True):
        self._stats = None
        self._norm_var = bool(norm_var)
        if rfilename and match(r'^(ark|scp)(,\w+)*:', rfilename):
            from pydrobert.kaldi import tables
            if key:
                with tables.open(rfilename, 'bm', 'r+') as stats_file:
                    self._stats = stats_file[key]
            else:
                # Assume its the first entry
                with tables.open(rfilename, 'bm') as stats_file:
                    self._stats = next(stats_file)
        elif rfilename:
            npy_stats = np.load(rfilename, fix_imports=True)
            if key:
                for entry in npy_stats:
                    if entry['key'] == key:
                        self._stats = entry['value'][:, :entry['num']]
                        break
                if self._stats is None:
                    raise KeyError(key)
            elif len(npy_stats.dtype):
                # assume a dictionary was stored here, but we weren't
                # given a key. Read the first one
                self._stats = npy_stats[0]['value'][:, :npy_stats[0]['num']]
            else:
                self._stats = npy_stats

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
        self._stats[0, :-1] += vec
        self._stats[1, :-1] += vec ** 2

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
        self._stats[0, :-1] += tensor.sum(axis=other_axes)
        self._stats[1, :-1] += (tensor ** 2).sum(axis=other_axes)

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
        if not in_place:
            vec = vec.copy()
        if self.have_stats:
            count = self._stats[0, -1]
            means = self._stats[0, :-1] / count
            if self._norm_var:
                varss = self._stats[1, :-1] / count - means ** 2
                close_zero = np.isclose(varss, 0)
                if np.any(close_zero):
                    warnings.warn('0 variance encountered. Replacing with 1')
                    varss[close_zero] = 1
                scales = 1 / np.sqrt(varss)
                offsets = -means * scales
            else:
                scales = np.ones(1)
                offsets = -means
            vec *= scales
            vec += offsets
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
        if not in_place:
            tensor = tensor.copy()
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
            scales = 1 / np.sqrt(varss)
            offsets = -means * scales
        else:
            scales = np.ones(1)
            offsets = -means
        tensor_slice = [None] * len(tensor.shape)
        tensor_slice[axis] = slice(None)
        tensor *= scales[tensor_slice]
        tensor += offsets[tensor_slice]
        return tensor

    def apply(self, features, axis=-1, in_place=False):
        if (features.shape and not np.prod(features.shape)) or \
                not len(features):
            raise ValueError('Cannot apply to empty array')
        if features.shape and len(features.shape) > 1:
            return self._apply_tensor(features, axis, in_place)
        else:
            return self._apply_vector(features, in_place)

    def save(self, wfilename, key=None):
        '''Save accumulated statistics to file

        If `wfilename` begins with ``r'^(ark|scp)(,\w+)*:'``, statistics
        will be saved as a Kaldi script/archive. Otherwise, statistics
        are saved in Numpy (``.npy``) format.

        If `key` is not set, statistics are stored directly to
        `wfilename`, overwriting anything that's already there.

        If `key` is set and the file is *not* a Kaldi script/archive,
        `save` will attempt to overwrite only the statistics for that
        key (if the file is written in key-value pairs). Kaldi
        scripts/archives will always be overwritten, since there is no
        way to check if it is safe to read them ahead of time (for
        example, "reading" stdout).

        Parameters
        ----------
        wfilename : str
        key : str, optional

        Raises
        ------
        ImportError
            If `wfilename` is a Kaldi table but `pydrobert.kaldi.tables`
            cannot be imported
        ValueError
            If no stats have been accumulated
        '''
        if not self.have_stats:
            raise ValueError('No stats have been accumulated to save')
        kaldi_table = match(r'^(ark|scp)(,\w+)*:', wfilename)
        if kaldi_table:
            from pydrobert.kaldi import tables
            if tables.KaldiDataType.BaseMatrix.is_double:
                kaldi_dtype = np.float64
            else:
                kaldi_dtype = np.float32
        if key is None:
            if kaldi_table:
                with tables.open(wfilename, 'bm', 'w') as file_obj:
                    file_obj.write(
                        Standardize.BOGUS_KEY,
                        self._stats.astype(kaldi_dtype),
                    )
            else:
                with open(wfilename, 'wb') as file_obj:
                    np.save(
                        file_obj,
                        self._stats,
                        fix_imports=True,
                    )
        else:
            all_stats = dict()
            if not kaldi_table and exists(wfilename):
                try:
                    past_stats_obj = np.load(wfilename, fix_imports=True)
                    if past_stats_obj.dtype.fields is None or \
                            'key' not in past_stats_obj.dtype.fields or \
                            'value' not in past_stats_obj.dtype.fields or \
                            'num' not in past_stats_obj.dtype.fields:
                        raise IOError()
                    for key_value in past_stats_obj:
                        key_2 = key_value['key']
                        value = key_value['value'][:key_value['num']]
                        all_stats[key_2] = value
                except IOError:
                    # were unable to load expected key/value object. No
                    # biggie
                    pass
            all_stats[key] = self._stats
            all_stats = list(all_stats.items())
            all_stats.sort()
            if kaldi_table:
                with tables.open(wfilename, 'bm', 'w') as file_obj:
                    for key, value in all_stats:
                        file_obj.write(key, value.astype(kaldi_dtype))
            else:
                max_key_len = 0
                max_num = 0
                for key, value in all_stats:
                    max_key_len = max(len(key), max_key_len)
                    max_num = max(value.shape[1], max_num)
                dt = np.dtype([
                    ('key', np.unicode, max_key_len),
                    ('value', np.float64, (2, max_num)),
                    ('num', np.uint16),
                ])
                stat_obj = np.empty(len(all_stats), dtype=dt)
                for idx, (key, value) in enumerate(all_stats):
                    stat_obj[idx]['key'] = key
                    stat_obj[idx]['value'][:, :value.shape[1]] = value
                    stat_obj[idx]['num'] = value.shape[1]
                with open(wfilename, 'wb') as file_obj:
                    np.save(
                        file_obj,
                        stat_obj,
                        fix_imports=True,
                    )

CMVN = Standardize

class Deltas(PostProcessor):
    '''Calculate feature deltas (weighted rolling averages)

    Deltas are calculated by correlating the feature tensor with a 1D
    delta filter (`Deltas.DELTA_FILTER`) along an axis (e.g. the feature
    frame axis). The delta filter is low-pass, spreading energy along
    the feature axis. Deltas can be cascaded to approximate higher order
    moments.

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

    Parameters
    ----------
    num_deltas : int
    target_axis : int, optional
    concatenate : bool, optional
    pad_mode : str or function, optional
        How to pad the input sequence when correlating. Additional
        keyword arguments will be passed to `numpy.pad`. See `numpy.pad`
        for more details
    '''

    DELTA_FILTER = np.array((-.2, -.1, 0., .1, .2), dtype=np.float64)
    '''The filter features are correlated with to calculate deltas'''

    def __init__(
            self, num_deltas, target_axis=-1, concatenate=True,
            pad_mode='edge', **kwargs):
        self._target_axis = target_axis
        self._pad_mode = pad_mode
        self._pad_kwargs = kwargs
        self._concatenate = bool(concatenate)
        self._filts = [np.ones(1, dtype=np.float64)]
        for idx in range(num_deltas):
            self._filts.append(
                np.convolve(self._filts[idx], Deltas.DELTA_FILTER))

    def apply(self, features, axis=-1, in_place=False):
        if not np.prod(features.shape):
            if not in_place:
                features = features.copy()
            new_shape = list(features.shape)
            if self._concatenate:
                new_shape[self._target_axis] *= len(self._filts)
            else:
                new_shape.insert(
                    self._target_axis % (len(features.shape) + 1),
                    len(self._filts),
                )
            return features.reshape(new_shape)
        if len(self._filts) == 1:
            if not in_place:
                features = features.copy()
            if self._concatenate:
                return features
            else:
                feat_slice = [slice(None) for _ in range(len(features.shape))]
                feat_slice.insert(
                    self._target_axis % (len(feat_slice) + 1), None)
                return features[feat_slice]
        delta_feats = [features]
        def _pad_lambda(z, h, l, r):
            z = np.pad(z, (l, r), self._pad_mode, **self._pad_kwargs)
            return np.correlate(z, h)
        for filt in self._filts[1:]:
            pad_left = len(filt) // 2
            pad_right = (len(filt) - 1) // 2
            filtered = np.apply_along_axis(
                _pad_lambda, axis, features, filt, pad_left, pad_right)
            assert filtered.shape == features.shape
            delta_feats.append(filtered)
        if self._concatenate:
            return np.concatenate(delta_feats, self._target_axis)
        else:
            return np.stack(delta_feats, self._target_axis)
