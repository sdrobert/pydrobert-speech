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

"""Miscellaneous utility functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import str as text
from re import match

import numpy as np

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

__all__ = [
    'gauss_quant',
    'hertz_to_angular',
    'angular_to_hertz',
    'circshift_fourier',
    'read_signal',
]


def _gauss_quant_odeh_evans(p, mu=0, std=1):
    r = 1 - p if p > .5 else p
    if r < 1e-20:
        z = 10
    else:
        y = (-2 * np.log(r)) ** .5
        z = ((
            (
                4.53642210148e-5 * y + .0204231210245
            ) * y + .342242088547) * y + 1) * y + .322232431088
        z /= (
            (
                (.0038560700634 * y + .10353775285) * y + .531103462366
            ) * y + .588581570495) * y + .099348462606
        z = y - z
    if p < .5:
        z = -z
    return z * std + mu


try:
    from scipy.stats import norm

    def gauss_quant(p, mu=0, std=1):
        return norm.ppf(p) * std + mu
except ImportError:
    gauss_quant = _gauss_quant_odeh_evans
gauss_quant.__doc__ = """\
Gaussian quantile function

Given a probability from a univariate Gaussian, determine the
value of the random variable such that the probability of
drawing a value l.t.e. to that value is equal to the
probability. In other words, the so-called inverse cumulative
distribution function.

If ``scipy`` can be imported, this function uses ``scipy.norm.ppf``
to calculate the result. Otherwise, it uses the approximation from
Odeh & Evans 1974 (thru Brophy 1985)

Parameters
----------
p : float
    The probability
mu : float
    The Gaussian mean
std : float
    The Gaussian standard deviation

Returns
-------
float
    The random variable value
"""


def hertz_to_angular(hertz, samp_rate):
    """Convert cycles/sec to radians/sec"""
    return hertz * 2 * np.pi / samp_rate


def angular_to_hertz(angle, samp_rate):
    """Convert radians/sec to cycles/sec"""
    return angle * samp_rate / (2 * np.pi)


def circshift_fourier(filt, shift, start_idx=0, dft_size=None, copy=True):
    r"""Circularly shift a filter in the time domain, from the fourier domain

    A simple application of the shift theorem

    .. math:: DFT(T_u x)[k] = DFT(x)[k] e^{-2i\pi k u}

    Where we set ``u = shift / dft_size``

    Parameters
    ----------
    filt : 1D array-like
        The filter, in the fourier domain
    shift : float
        The number of samples to be translated by.
    start_idx : int, optional
        If `filt` is a truncated frequency response, this parameter
        indicates at what index in the dft the nonzero region starts
    dft_size : int, optional
        The dft_size of the filter. Defaults to
        ``len(filt) + start_idx``
    copy : bool, optional
        Whether it is okay to modify and return `filt`

    Returns
    -------
    array-like of complex128
        The filter frequency response, shifted by `u`
    """
    shift %= dft_size
    if dft_size is None:
        dft_size = len(filt) + start_idx
    if copy or filt.dtype != np.complex128:
        return filt * np.exp(-2j * np.pi * shift / dft_size * (
            np.arange(
                start_idx, start_idx + len(filt),
            ) % dft_size))
    else:
        filt *= np.exp(-2j * np.pi * shift / dft_size * (
            np.arange(
                start_idx, start_idx + len(filt),
            ) % dft_size))
        return filt


def _kaldi_table_read_signal(rfilename, dtype, key, **kwargs):
    from pydrobert.kaldi.io import open as io_open
    if key is None:
        key = 0
    if dtype is None:
        dtype = 'bm'
    if isinstance(key, str) or isinstance(key, text):
        with io_open(rfilename, dtype, mode='r+', **kwargs) as table:
            return table[key]
    else:
        with io_open(rfilename, dtype, mode='r', **kwargs) as table:
            for idx in range(key):
                if not table.move():
                    raise IndexError('table index out of range')
            return table.value()


def _scipy_io_read_signal(rfilename, dtype, key, **kwargs):
    from scipy.io import wavfile
    _, data = wavfile.read(rfilename, **kwargs)
    if dtype:
        data = data.astype(dtype)
    return data


def _wave_read_signal(rfilename, dtype, key, **kwargs):
    import wave
    wave_file = wave.open(rfilename, **kwargs)
    try:
        dtype_in = '<i{}'.format(wave_file.getsampwidth())
        data = np.frombuffer(
            wave_file.readframes(wave_file.getnframes()),
            dtype=dtype_in
        )
        n_data_points = len(data)
        n_channels = wave_file.getnchannels()
        if n_data_points % n_channels:
            raise IOError(
                'Number of channels do not evenly divide wave samples')
        if n_channels > 1:
            data = data.reshape(
                (n_data_points // n_channels, n_channels), order='C')
    finally:
        wave_file.close()
    if dtype:
        data = data.astype(dtype)
    return data


def _hdf5_read_signal(rfilename, dtype, key, **kwargs):
    import h5py
    with h5py.File(rfilename, 'r', **kwargs) as h5py_file:
        if key:
            data = h5py_file[key]
        else:
            group_stack = [h5py_file]
            data = None
            while group_stack:
                cur_group = group_stack.pop()
                if isinstance(cur_group, h5py.Dataset):
                    data = cur_group
                    break
                else:
                    keys = list(cur_group.keys())
                    keys.sort(reverse=True)
                    for name in keys:
                        group_stack.append(cur_group[name])
            if data is None:
                raise IOError('Could not find any dataset')
        if dtype:
            data = np.array(data, dtype=dtype)
        else:
            data = np.array(data)
    return data


def _numpy_binary_read_signal(rfilename, dtype, key, **kwargs):
    data = np.load(rfilename, **kwargs)
    if dtype:
        data = data.astype(dtype)
    return data


def _numpy_archive_read_signal(rfilename, dtype, key, **kwargs):
    archive = np.load(rfilename, **kwargs)
    if key:
        data = archive[key]
    else:
        data = archive['arr_0']
    if dtype:
        data = data.astype(dtype)
    return data


def _torch_read_signal(rfilename, dtype, key, **kwargs):
    import torch
    data = torch.load(rfilename, map_location='cpu', **kwargs).numpy()
    if dtype:
        data = data.astype(dtype)
    return data


def _kaldi_input_read_signal(rfilename, dtype, key, **kwargs):
    from pydrobert.kaldi.io import open as io_open
    if dtype is None:
        dtype = 'bm'
    with io_open(rfilename, mode='r', **kwargs) as inp_stream:
        data = inp_stream.read(dtype)
    return data


def _numpy_fromfile_read_signal(rfilename, dtype, key, **kwargs):
    if dtype:
        data = np.fromfile(rfilename, dtype=dtype, **kwargs)
    else:
        data = np.fromfile(rfilename, **kwargs)
    return data


def read_signal(rfilename, dtype=None, key=None, force_as=None, **kwargs):
    r"""Read a signal from a variety of possible sources

    Though the goal of this function is to return an array representing
    a signal of some sort, the way it goes about doing so depends on
    the setting of `rfilename`, processed in the following order:

    1. If `rfilename` starts with the regular expression
       ``r'^(ark|scp)(,\w+)*:'``, the file is treated as a Kaldi table
       and opened with the kaldi data type `dtype` (defaults to
       `BaseMatrix`). The package :mod:`pydrobert.kaldi` will be imported
       to handle reading. If `key` is set, the value associated with
       that key is retrieved. Otherwise the first listed value is
       returned.
    2. If `rfilename` ends with ``.wav``, the file is assumed to be a
       wave file. The function will rely on the :mod:`scipy` package to load
       the file if :mod:`scipy` can be imported. Otherwise, it uses the
       standard :mod:`wave` package. The type of data encodings each package
       can handle varies, though neither can handle compressed data.
    3. If `rfilename` ends with ``.hdf5``, the file is assumed to be an
       HDF5 file. HDF5 and :mod:`h5py` must be installed on the host system to
       read this way. If `key` is set, the data will assumed to be
       indexed by `key` on the archive. Otherwise, a depth-first search
       of the archive will be performed for the first data set. If set,
       data will be cast to as the numpy data type `dtype`
    4. If `rfilename` ends with ``.npy``, the file is assumed to be a
       binary in Numpy format. If set, the result will be cast as
       the numpy data type `dtype`.
    5. If `rfilename` ends with ``.npz``, the file is assumed to be an
       archive in numpy format. If `key` is swet, the data indexed by
       `key` will be loaded. Otherwise the data indexed by the key
       ``'arr_0'`` will be loaded. If set, the result will be cast as
       the numpy data type `dtype`.
    6. If `rfilename` ends with ``.pt``, the file is assumed to be a binary
       in PyTorch format. If set, the results will be cast as the numpy
       data type `dtype`.
    7. If `rfilename` ends with ``.sph``, the file is assumed to be a NIST
       SPHERE file. If set, the results will be cast as the numpy data type
       `dtype`
    8. If :mod:`pydrobert.kaldi` can be imported, it will try to read an
       object of kaldi data type `dtype` (defaults to ``BaseMatrix``)
       from a basic kaldi input stream. If this fails, we continue
       to step 9.
    9. Otherwise, the routine :func:`numpy.fromfile` will be used to load the
       data (of type `dtype`, if provided). :func:`numpy.tofile` does not
       keep track of shape data, so any read data will be 1D.

    Additional keyword arguments are passed along to the associated
    open or read operation.

    Parameters
    ----------
    rfilename : str
    dtype : object, optional
    key : object, optional
    force_as : {
                None, 'tab', 'wav', 'hdf5', 'npy', 'npz', 'pt', 'sph', 'kaldi',
                'file',
            }, optional
        If not :obj:`None`, forces `rfilename` to be interpreted as a specific
        file type, bypassing the above selection strategy. ``'tab'``: Kaldi
        table; ``'wav'``: wave file; ``'hdf5'``: HDF5 file; ``'npy'``: Numpy
        binary; ``'npz'``: Numpy archive; ``'pt'``: PyTorch binary; ``'sph'``:
        NIST sphere; ``'kaldi'`` Kaldi object; ``'file'`` read via
        :func:`numpy.fromfile`

    Returns
    -------
    array-like

    Notes
    -----

    Code for reading SPHERE files was based off of `sph2pipe v 2.5
    <https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools>`__.
    That code can only suppport the "shorten" audio format up to version 2.
    """
    if force_as is None:
        if match(r'^(ark|scp)(,\w+)*:', rfilename):
            force_as = 'table'
        elif rfilename.endswith('.wav'):
            force_as = 'wav'
        elif rfilename.endswith('.hdf5'):
            force_as = 'hdf5'
        elif rfilename.endswith('.npy'):
            force_as = 'npy'
        elif rfilename.endswith('.npz'):
            force_as = 'npz'
        elif rfilename.endswith('.pt'):
            force_as = 'pt'
        elif rfilename.endswith('.sph'):
            force_as = 'sph'
    if force_as == 'table':
        data = _kaldi_table_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == 'wav':
        try:
            data = _scipy_io_read_signal(rfilename, dtype, key, **kwargs)
        except ImportError:
            data = _wave_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == 'hdf5':
        data = _hdf5_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == 'npy':
        data = _numpy_binary_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == 'npz':
        data = _numpy_archive_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == 'pt':
        data = _torch_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == 'sph':
        from ._sphere import sphere_read_signal
        data = sphere_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == 'kaldi':
        data = _kaldi_input_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == 'file':
        data = _numpy_fromfile_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as is None:
        try:
            data = _kaldi_input_read_signal(rfilename, dtype, key, **kwargs)
        except Exception:
            data = _numpy_fromfile_read_signal(rfilename, dtype, key, **kwargs)
    else:
        raise ValueError(
            'force_as ({}) is not one of table, wav, hdf5, npy, npz, pt, sph,'
            'kaldi, file'.format(force_as))
    return data


def alias_factory_subclass_from_arg(factory_class, arg):
    '''Boilerplate for getting an instance of an AliasedFactory

    Rather than an instance itself, a function could receive the
    arguments to initialize an AliasedFactory with ``from_alias``.
    This function uses the following strategy to try and do so::

    1. If ``arg`` is an instance of ``factory_class``, return ``arg``
    2. If ``arg`` is a string, use it as the alias
    3. a. Copy ``arg`` to a dictionary
       b. Pop the key ``'alias'`` and treat the rest as keyword arguments
       c. If the key ``'alias'`` is not found, try ``'name'``

    This function is intentionally limited in order to work nicely with
    JSON config files.
    '''
    if isinstance(arg, factory_class):
        return arg
    elif isinstance(arg, str) or isinstance(arg, text):
        return factory_class.from_alias(arg)
    else:
        arg = dict(arg)
        try:
            alias = arg.pop('alias')
        except KeyError:
            alias = arg.pop('name')
        return factory_class.from_alias(alias, **arg)
