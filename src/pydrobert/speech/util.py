# Copyright 2021 Sean Robertson

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

import warnings

from re import match
from typing import Any, Optional

import pydrobert.speech.config as config
import numpy as np

from pydrobert.speech.alias import alias_factory_subclass_from_arg as _afsfa


__all__ = [
    "angular_to_hertz",
    "circshift_fourier",
    "gauss_quant",
    "hertz_to_angular",
    "read_signal",
]


def alias_factory_subclass_from_arg(*args, **kwargs):
    warnings.warn(
        "using alias_factory_subclass_from_arg from util is deprecated. "
        "Use from pydrobert.speech.alias instead",
        category=DeprecationWarning,
        stacklevel=2,
    )

    return _afsfa(*args, **kwargs)


def _gauss_quant_odeh_evans(p: float, mu: float = 0, std: float = 1) -> float:
    r = 1 - p if p > 0.5 else p
    if r < 1e-20:
        z = 10
    else:
        y = (-2 * np.log(r)) ** 0.5
        z = (
            ((4.53642210148e-5 * y + 0.0204231210245) * y + 0.342242088547) * y + 1
        ) * y + 0.322232431088
        z /= (
            ((0.0038560700634 * y + 0.10353775285) * y + 0.531103462366) * y
            + 0.588581570495
        ) * y + 0.099348462606
        z = y - z
    if p < 0.5:
        z = -z
    return z * std + mu


try:
    from scipy.stats import norm

    def gauss_quant(p: float, mu: float = 0, std: float = 1) -> float:
        return norm.ppf(p) * std + mu


except ImportError:
    gauss_quant = _gauss_quant_odeh_evans
gauss_quant.__doc__ = """\
Gaussian quantile function

Given a probability from a univariate Gaussian, determine the value of the random
variable such that the probability of drawing a value l.t.e. to that value is equal to
the probability. In other words, the so-called inverse cumulative distribution function.

If `scipy` can be imported, this function uses :func:`scipy.norm.ppf` to calculate the
result. Otherwise, it uses the approximation from Odeh & Evans 1974 (thru Brophy 1985)

Parameters
----------
p
    The probability
mu
    The Gaussian mean
std
    The Gaussian standard deviation

Returns
-------
q : float
    The random variable value
"""


def hertz_to_angular(hertz: float, samp_rate: float) -> float:
    """Convert cycles/sec to radians/sec"""
    return hertz * 2 * np.pi / samp_rate


def angular_to_hertz(angle: float, samp_rate: float) -> float:
    """Convert radians/sec to cycles/sec"""
    return angle * samp_rate / (2 * np.pi)


def circshift_fourier(
    filt: np.ndarray,
    shift: float,
    start_idx: int = 0,
    dft_size: int = None,
    copy: bool = True,
) -> np.ndarray:
    r"""Circularly shift a filter in the time domain, from the fourier domain

    A simple application of the shift theorem

    .. math::

        DFT(T_u x)[k] = DFT(x)[k] e^{-2i\pi k u}

    Where we set ``u = shift / dft_size``

    Parameters
    ----------
    filt
        The filter, in the fourier domain
    shift
        The number of samples to be translated by.
    start_idx
        If `filt` is a truncated frequency response, this parameter indicates at what
        index in the dft the nonzero region starts
    dft_size
        The dft_size of the filter. Defaults to
        ``len(filt) + start_idx``
    copy
        Whether it is okay to modify and return `filt`

    Returns
    -------
    out : np.ndarray
        The 128-bit complex filter frequency response, shifted by `u`
    """
    shift %= dft_size
    if dft_size is None:
        dft_size = len(filt) + start_idx
    if copy or filt.dtype != np.complex128:
        return filt * np.exp(
            -2j
            * np.pi
            * shift
            / dft_size
            * (np.arange(start_idx, start_idx + len(filt),) % dft_size)
        )
    else:
        filt *= np.exp(
            -2j
            * np.pi
            * shift
            / dft_size
            * (np.arange(start_idx, start_idx + len(filt),) % dft_size)
        )
        return filt


def _kaldi_table_read_signal(rfilename, dtype, key, **kwargs):
    from pydrobert.kaldi.io import open as io_open  # type: ignore

    if key is None:
        key = 0
    if dtype is None:
        dtype = "bm"
    if isinstance(key, str):
        with io_open(rfilename, dtype, mode="r+", **kwargs) as table:
            return table[key]
    else:
        with io_open(rfilename, dtype, mode="r", **kwargs) as table:
            for idx in range(key):
                if not table.move():
                    raise IndexError("table index out of range")
            return table.value()


def _scipy_io_read_signal(rfilename, dtype, key, **kwargs):
    from scipy.io import wavfile  # type: ignore

    _, data = wavfile.read(rfilename, **kwargs)
    if dtype:
        data = data.astype(dtype)
    return data


def _wave_read_signal(rfilename, dtype, key, **kwargs):
    import wave

    wave_file = wave.open(rfilename, **kwargs)
    try:
        dtype_in = "<i{}".format(wave_file.getsampwidth())
        data = np.frombuffer(
            wave_file.readframes(wave_file.getnframes()), dtype=dtype_in
        )
        n_data_points = len(data)
        n_channels = wave_file.getnchannels()
        if n_data_points % n_channels:
            raise IOError("Number of channels do not evenly divide wave samples")
        if n_channels > 1:
            data = data.reshape((n_data_points // n_channels, n_channels), order="C")
    finally:
        wave_file.close()
    if dtype:
        data = data.astype(dtype)
    return data


def _hdf5_read_signal(rfilename, dtype, key, **kwargs):
    import h5py  # type: ignore

    with h5py.File(rfilename, "r", **kwargs) as h5py_file:
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
                raise IOError("Could not find any dataset")
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
        data = archive["arr_0"]
    if dtype:
        data = data.astype(dtype)
    return data


def _torch_read_signal(rfilename, dtype, key, **kwargs):
    import torch  # type: ignore

    data = torch.load(rfilename, map_location="cpu", **kwargs).numpy()
    if dtype:
        data = data.astype(dtype)
    return data


def _kaldi_input_read_signal(rfilename, dtype, key, **kwargs):
    from pydrobert.kaldi.io import open as io_open  # type: ignore

    if dtype is None:
        dtype = "bm"
    with io_open(rfilename, mode="r", **kwargs) as inp_stream:
        data = inp_stream.read(dtype)
    return data


def _numpy_fromfile_read_signal(rfilename, dtype, key, **kwargs):
    if dtype:
        data = np.fromfile(rfilename, dtype=dtype, **kwargs)
    else:
        data = np.fromfile(rfilename, **kwargs)
    return data


def _soundfile_read_signal(rfilename, dtype, key, **kwargs):
    import soundfile

    fi = soundfile.info(rfilename)
    if fi.subtype == "FLOAT":
        dtype_ = np.float32
    elif fi.subtype == "DOUBLE":
        dtype_ = np.float64
    elif fi.subtype == "PCM_S8":
        dtype_ = np.int8
    elif fi.subtype == {"PCM_U8"}:
        dtype_ = np.uint8
    elif fi.subtype in {"PCM_32", "PCM_24"}:
        dtype_ = np.int32
    else:
        # FIXME(sdrobert): PCM_16 is a decent guess for the remainder of types, but
        # it's definitely not complete
        dtype_ = np.int16
    data = soundfile.read(rfilename, dtype=dtype_, **kwargs)[0]
    if dtype is not None:
        # if you don't do this as a second stage and you want floats out the back,
        # soundfile will scale those to the range +/- 1. Other decoders are two-stage
        # as well.
        data = data.astype(dtype)
    return data


def read_signal(
    rfilename: str,
    dtype: Optional[np.dtype] = None,
    key: Any = None,
    force_as: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    r"""Read a signal from a variety of possible sources

    Though the goal of this function is to return an array representing a signal of some
    sort, the way it goes about doing so depends on the setting of `rfilename`,
    processed in the following order:

    1.  If `rfilename` starts with the regular expression :obj:`r'^(ark|scp)(,\w+)*:'`,
        the file is treated as a Kaldi table and opened with the kaldi data type `dtype`
        (defaults to :class:`BaseMatrix`). The package :mod:`pydrobert.kaldi` will be
        imported to handle reading. If `key` is set, the value associated with that key
        is retrieved. Otherwise the first listed value is returned.
    2.  If `rfilename` ends with a file type listed in
        :obj:`pydrobert.speech.config.SOUNDFILE_SUPPORTED_FILE_TYPES` (requires
        :mod:`soundfile`), the file will be opened with that audio file type.
    3.  If `rfilename` ends with :obj:`'.wav'`, the file is assumed to be a wave file.
        The function will rely on the :mod:`scipy` package to load the file if
        :mod:`scipy` can be imported. Otherwise, it uses the standard :mod:`wave`
        package. The type of data encodings each package can handle varies, though
        neither can handle compressed data.
    4.  If `rfilename` ends with :obj:`'.hdf5'`, the file is assumed to be an HDF5 file.
        HDF5 and :mod:`h5py` must be installed on the host system to read this way. If
        `key` is set, the data will assumed to be indexed by `key` on the archive.
        Otherwise, a depth-first search of the archive will be performed for the first
        data set. If set, data will be cast to as the numpy data type `dtype`
    5.  If `rfilename` ends with :obj:`'.npy'`, the file is assumed to be a binary in
        Numpy format. If set, the result will be cast as the numpy data type `dtype`.
    6.  If `rfilename` ends with :obj:`'.npz'`, the file is assumed to be an archive in
        Numpy format. If `key` is swet, the data indexed by `key` will be loaded.
        Otherwise the data indexed by the key :obj:`'arr_0'` will be loaded. If set, the
        result will be cast as the numpy data type `dtype`.
    7.  If `rfilename` ends with :obj:`'.pt'`, the file is assumed to be a binary in
        PyTorch format. If set, the results will be cast as the numpy data type `dtype`.
    8.  If `rfilename` ends with :obj:`'.sph'`, the file is assumed to be a NIST SPHERE
        file. If set, the results will be cast as the numpy data type `dtype`
    9.  If `rfilename`` ends with ``'|'``, it will try to read an object of kaldi data
        type `dtype` (defaults to :class:`BaseMatrix`) from a basic kaldi input stream.
    10. Otherwise, we throw an :class:`IOError`

    Additional keyword arguments are passed along to the associated
    open or read operation.

    Parameters
    ----------
    rfilename 
    dtype
    key
    force_as
        If not :obj:`None`, forces `rfilename` to be interpreted as a specific
        file type, bypassing the above selection strategy. ``'tab'``: Kaldi
        table; ``'wav'``: wave file; ``'hdf5'``: HDF5 file; ``'npy'``: Numpy
        binary; ``'npz'``: Numpy archive; ``'pt'``: PyTorch binary; ``'sph'``:
        NIST sphere; ``'kaldi'`` Kaldi object; ``'file'`` read via
        :func:`numpy.fromfile`. The types in :obj:`SOUNDFILE_SUPPORTED_FILE_TYPES`
        are also valid values. `'soundfile'` will use :mod:`soundfile` to read the file
        regardless of the suffix.
    **kwargs

    Returns
    -------
    signal : np.ndarray

    Warnings
    --------
    Post v 0.2.0, the behaviour after step 8 changed. Instead of trying to read first as
    Kaldi input, and, failing that, via :func:`numpy.fromfile`, we try to read as Kaldi
    input if the file name ends with ``'|'`` and error otherwise. The catch-all
    behaviour was disabled due to the interaction with
    :obj:`pydrobert.speech.config.SOUNDFILE_SUPPORTED_FILE_TYPES` whose value depends on
    the existence of :mod:`soundfile` and the underlying version of `libsndfile
    <https://libsndfile.github.io/libsndfile>`__.

    Notes
    -----
    Python code for reading SPHERE files (not via :mod:soundfile`) was based off of
    `sph2pipe v 2.5
    <https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools>`__.
    That code can only suppport the "shorten" audio format up to version 2.
    """
    if force_as is None:
        if match(r"^(ark|scp)(,\w+)*:", rfilename):
            force_as = "table"
        elif (
            rfilename.rsplit(".", maxsplit=1)[-1]
            in config.SOUNDFILE_SUPPORTED_FILE_TYPES
        ):
            force_as = "soundfile"
        elif rfilename.endswith(".wav"):
            force_as = "wav"
        elif rfilename.endswith(".hdf5"):
            force_as = "hdf5"
        elif rfilename.endswith(".npy"):
            force_as = "npy"
        elif rfilename.endswith(".npz"):
            force_as = "npz"
        elif rfilename.endswith(".pt"):
            force_as = "pt"
        elif rfilename.endswith(".sph"):
            force_as = "sph"
        elif rfilename.endswith("|"):
            force_as = "kaldi"
        else:
            raise IOError(f"Unable to infer file type from {rfilename}. Set force_as.")
    if force_as == "table":
        data = _kaldi_table_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == "wav":
        try:
            data = _scipy_io_read_signal(rfilename, dtype, key, **kwargs)
        except ImportError:
            data = _wave_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == "hdf5":
        data = _hdf5_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == "npy":
        data = _numpy_binary_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == "npz":
        data = _numpy_archive_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == "pt":
        data = _torch_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == "sph":
        from ._sphere import sphere_read_signal

        data = sphere_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == "kaldi":
        data = _kaldi_input_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == "file":
        data = _numpy_fromfile_read_signal(rfilename, dtype, key, **kwargs)
    elif force_as == "soundfile":
        data = _soundfile_read_signal(rfilename, dtype, key, **kwargs)
    else:
        avail_force_as = {
            "table",
            "wav",
            "hdf5",
            "npy",
            "npz",
            "pt",
            "sph",
            "kaldi",
            "file",
            "soundfile",
        } | config.SOUNDFILE_SUPPORTED_FILE_TYPES
        msg = f"force_as ('{force_as}') is not one of {avail_force_as}."
        if force_as in config._BASE_SOUNDFILE_SUPPORTED_TYPES:
            msg += (
                "\n... but it could be, with the proper version of libsndfile and "
                "pysoundfile installed"
            )
        elif force_as in config._FULL_SOUNDFILE_SUPPORTED_TYPES:
            msg += (
                "\n... but pysoundfile may be able to handle it. "
                "Try setting force_as = 'soundfile'"
            )
        raise ValueError(msg)
    return data

