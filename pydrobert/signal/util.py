"""Miscellaneous utility functions"""

import numpy as np

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

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
