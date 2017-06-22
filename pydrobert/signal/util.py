"""Miscellaneous utility functions"""

from re import match

import numpy as np

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

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
gauss_quant.__doc__ = \
"""Gaussian quantile function

Given a probability from a univariate Gaussian, determine the
value of the random variable such that the probability of
drawing a value l.t.e. to that value is equal to the
probability. In other words, the so-called inverse cumulative
distribution function.

If `scipy` can be imported, this function uses `scipy.norm.ppf`
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

    A simple application of the shift theorem:

    .. math:: DFT(T_u x)_k = DFT(x)_k e^{-2i\pi k shift / dft_size}

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

def _kaldi_read_signal(rfilename, dtype, key, **kwargs):
    from pydrobert.kaldi import tables
    if key is None:
        key = 0
    if dtype is None:
        dtype = 'bm'
    if isinstance(key, str):
        with tables.open(rfilename, dtype, mode='r+', **kwargs) as table:
            return table[key]
    else:
        with tables.open(rfilename, dtype, mode='r', **kwargs) as table:
            for idx in range(key):
                if not table.move():
                    raise IndexError('table index out of range')
            return table.value

def _scipy_io_read_signal(rfilename, dtype, key, **kwargs):
    from scipy.io import wavfile
    if key is not None:
        raise TypeError("'key' is an invalid keyword argument for wave files")
    rate, data = wavfile.read(rfilename, **kwargs)
    if dtype:
        data = data.astype(dtype)
    return data

def _wave_read_signal(rfilename, dtype, key, **kwargs):
    import wave
    if key is not None:
        raise TypeError("'key' is an invalid keyword argument for wave files")
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
    with h5py.File(rfilename, **kwargs) as h5py_file:
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
                    for name in cur_group.keys()[::-1]:
                        group_stack.append(cur_group[name])
            if data is None:
                raise IOError('Could not find any dataset')
        if dtype:
            data = np.array(data, dtype=dtype)
        else:
            data = np.array(data)
    return data

def _numpy_binary_read_signal(rfilename, dtype, key, **kwargs):
    if key is not None:
        raise TypeError(
            "'key' is an invalid keyword argument for numpy binaries")
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

def _numpy_fromfile_read_signal(rfilename, dtype, key, **kwargs):
    if key is not None:
        raise TypeError(
            "'key' is an invalid keyword argument for fromfile binaries")
    if dtype:
        data = np.fromfile(rfilename, dtype=dtype, **kwargs)
    else:
        data = np.fromfile(rfilename, **kwargs)
    return data

def read_signal(rfilename, dtype=None, key=None, **kwargs):
    r"""Read a signal from a variety of possible sources

    Though the goal of this function is to return an array representing
    a signal of some sort, the way it goes about doing so depends on
    the setting of `rfilename`, processed in the following order:

    If `rfilename` starts with the regular expression
    ``r'^(ark|scp)(,\w+)*:'``, the file is treated as a Kaldi
    archive/script. The package `pydrobert.kaldi` will be imported to
    handle reading.

    If `rfilename` ends with `.wav`, the file is assumed to be a wave
    file. The function will rely on the `scipy` package to load the
    file if `scipy` can be imported. Otherwise, it uses the standard
    `wave` package. The type of data encondings each package can handle
    varies, though neither can handle compressed data.

    If `rfilename` ends with `.hdf5`, the file is assumed to be an
    HDF5 file. HDF5 and h5py must be installed on the host system to
    read this way.

    If `rfilename` ends with `.npy`, the file is assumed to be a binary
    in Numpy format.

    If `rfilename` ends with `.npz`, the file is assumed to be an
    archive in numpy format.

    Otherwise, the routine `numpy.fromfile` will be used to load the
    data. `numpy.tofile` does not keep track of shape data, so any
    read data will be 1D.

    Additional keyword arguments are passed along to the associated
    open or read operation.

    Parameters
    ----------
    rfilename : str
    dtype : object, optional
        For Kaldi archives/scripts, this is a value in the enum
        `pydrobert.kaldi.tables.KaldiDataType` (if not provided, the
        base matrix is assumed). If `numpy.fromfile` is used, it is
        passed as a keyword argument to the routine (default is float).
        Otherwise, the resulting array will be cast to `dtype` before
        returning
    key : object, optional
        For Kaldi archives/scripts, `key` is used to retrieve the data
        as the corresponding value: a `key` of type `str` is assumed to
        be a random access key, while a `key` of type `int` is assumed
        to index the value in a sequential read. For HDF5 files, `key`
        is used to access the dataset with square brackets from the
        root `h5py.File` object. Likewise for a numpy archive. Other
        read styles will raise a `TypeError` if `key` is set. The
        default behaviour for read styles with keys is to take the first
        data set available. In the case of an HDF5 file, this is through
        a depth-first search

    Returns
    -------
    array-like

    Raises
    ------
    ImportError
    TypeError
    IOError
    """
    if match(r'^(ark|scp)(,\w+)*:', rfilename):
        data = _kaldi_read_signal(rfilename, dtype, key, **kwargs)
    elif rfilename.endswith('.wav'):
        try:
            data = _scipy_io_read_signal(rfilename, dtype, key, **kwargs)
        except ImportError:
            data = _wave_read_signal(rfilename, dtype, key, **kwargs)
    elif rfilename.endswith('.hdf5'):
        data = _hdf5_read_signal(rfilename, dtype, key, **kwargs)
    elif rfilename.endswith('.npy'):
        data = _numpy_binary_read_signal(rfilename, dtype, key, **kwargs)
    elif rfilename.endswith('.npz'):
        data = _numpy_archive_read_signal(rfilename, dtype, key, **kwargs)
    else:
        data = _numpy_fromfile_read_signal(rfilename, dtype, key, **kwargs)
    return data
