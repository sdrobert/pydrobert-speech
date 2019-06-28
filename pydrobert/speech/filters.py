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

"""Filters and filter banks"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np

from pydrobert.speech import AliasedFactory
from pydrobert.speech import config
from pydrobert.speech.scales import MelScaling
from pydrobert.speech.scales import ScalingFunction
from pydrobert.speech.util import alias_factory_subclass_from_arg
from pydrobert.speech.util import angular_to_hertz
from pydrobert.speech.util import hertz_to_angular

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

__all__ = [
    'LinearFilterBank',
    'TriangularOverlappingFilterBank',
    'GaborFilterBank',
    'ComplexGammatoneFilterBank',
    'WindowFunction',
    'BartlettWindow',
    'BlackmanWindow',
    'HammingWindow',
    'HannWindow',
    'GammaWindow',
]

# banks


class LinearFilterBank(AliasedFactory):
    """A collection of linear, time invariant filters

    A ``LinearFilterBank`` instance is expected to provide factory
    methods for instantiating a fixed number of LTI filters in either
    the time or frequency domain. Filters should be organized lowest
    frequency first.

    Attributes
    ----------
    is_real : bool
    is_analytic : bool
    is_zero_phase : bool
    num_filts : int
    sampling_rate : float
    centers_hz : tuple
    supports_hz : tuple
    supports : tuple
    supports_ms : tuple
    """

    @abc.abstractproperty
    def is_real(self):
        """Whether the filters are real or complex"""
        pass

    @abc.abstractproperty
    def is_analytic(self):
        """Whether the filters are (approximately) analytic"""
        pass

    @abc.abstractproperty
    def is_zero_phase(self):
        """Whether the filters are zero phase or not

        Zero phase filters are even functions with no imaginary part
        in the fourier domain. Their impulse responses center around 0.
        """
        pass

    @abc.abstractproperty
    def num_filts(self):
        """Number of filters in the bank"""
        pass

    @abc.abstractproperty
    def sampling_rate(self):
        """Number of samples in a second of a target recording"""
        pass

    @abc.abstractproperty
    def supports_hz(self):
        """Boundaries of effective support of filter freq responses, in Hz.

        Returns a tuple of length `num_filts` containing pairs of floats
        of the low and high frequencies. Frequencies outside the span
        have a response of approximately (with magnitude up to
        `speech.EFFECTIVE_SUPPORT_SIGNAL`) zero.

        The boundaries need not be tight, i.e. the region inside the
        boundaries could be zero. It is more important to guarantee that
        the region outside the boundaries is approximately zero.

        The boundaries ignore the Hermitian symmetry of the filter if it
        is real. Bounds of ``(10, 20)`` for a real filter imply that the
        region ``(-20, -10)`` could also be nonzero.

        The user is responsible for adjusting the for the periodicity
        induced by sampling. For example, if the boundaries are
        ``(-5, 10)`` and the filter is sampled at 15Hz, then all bins
        of an associated DFT could be nonzero.
        """
        pass

    @abc.abstractproperty
    def supports(self):
        """Boundaries of effective support of filter impulse resps, in samples

        Returns a tuple of length `num_filts` containing pairs of
        integers of the first and last (effectively) nonzero samples.

        The boundaries need not be tight, i.e. the region inside the
        boundaries could be zero. It is more important to guarantee that
        the region outside the boundaries is approximately zero.

        If a filter is instantiated using a buffer that is unable to
        fully contain the supported region, samples will wrap around the
        boundaries of the buffer.

        Noncausal filters will have start indices less than 0. These
        samples will wrap to the end of the filter buffer when the
        filter is instantiated.
        """
        pass

    @property
    def supports_ms(self):
        """Boundaries of effective support of filter impulse resps, in ms"""
        return tuple(
            (
                s[0] * 1000 / self.sampling_rate,
                s[1] * 1000 / self.sampling_rate,
            ) for s in self.supports)

    @abc.abstractmethod
    def get_impulse_response(self, filt_idx, width):
        """Construct filter impulse response in a fixed-width buffer

        Construct the filter in the time domain.

        Parameters
        ----------
        filt_idx : int
            The index of the filter to generate. Less than `num_filts`
        width : int
            The length of the buffer, in samples. If less than the
            support of the filter, the filter will alias.

        Returns
        -------
        array-like
            1D float64 or complex128 numpy array of length `width`
        """
        pass

    @abc.abstractmethod
    def get_frequency_response(self, filt_idx, width, half=False):
        """Construct filter frequency response in a fixed-width buffer

        Construct the 2pi-periodized filter in the frequency domain.
        Zero-phase filters `is_zero_phase` are returned as 8-byte float
        arrays. Otherwise, they will be 16-byte complex floats.

        Parameters
        ----------
        filt_idx : int
            The index of the filter to generate. Less than `num_filts`
        width : int
            The length of the DFT to output
        half : bool, optional
            Whether to return only the DFT bins between [0,pi]

        Results
        -------
        array-like
            If `half` is `False`, returns a 1D float64 or complex128
            numpy array of length `width`. If `half` is `True` and
            `width` is even, the returned array is of length
            ``width // 2 + 1``. If `width` is odd, the returned array
            is of length ``(width + 1) // 2``.
        """
        pass

    @abc.abstractmethod
    def get_truncated_response(self, filt_idx, width):
        """Get nonzero region of filter frequency response

        Many filters will be compactly supported in frequency (or
        approximately so). This method generates a tuple `(bin_idx,
        buf)` of the nonzero region.

        In the case of a complex filter, ``bin_idx + len(buf)`` may be
        greater than `width`; the filter wraps around in this case. The
        full frequency response can be calculated from the truncated
        response by:

        >>> bin_idx, trnc = bank.get_truncated_response(filt_idx, width)
        >>> full = numpy.zeros(width, dtype=trnc.dtype)
        >>> wrap = min(bin_idx + len(trnc), width) - bin_idx
        >>> full[bin_idx:bin_idx + wrap] = trnc[:wrap]
        >>> full[:len(trnc) - wrap] = tnc[wrap:]

        In the case of a real filter, only the nonzero region between
        ``[0, pi]`` (half-spectrum) is returned. No wrapping can occur
        since it would inevitably interfere with itself due to conjugate
        symmetry. The half-spectrum can easily be recovered by:

        >>> half_width = (width + width % 2) // 2 + 1 - width % 2
        >>> half = numpy.zeros(half_width, dtype=trnc.dtype)
        >>> half[bin_idx:bin_idx + len(trnc)] = trnc

        And the full spectrum by:

        >>> full[bin_idx:bin_idx + len(trnc)] = trnc
        >>> full[width - bin_idx - len(trnc) + 1:width - bin_idx + 1] = \\
        ...     trnc[:None if bin_idx else 0:-1].conj()

        (the embedded if-statement is necessary when bin_idx is 0, as
        the full fft excludes its symmetric bin)

        Parameters
        ----------
        filt_idx : int
            The index of the filter to generate. Less than `num_filts`
        width : int
            The length of the DFT to output

        Returns
        -------
        tuple of int, array

        """
        pass


class TriangularOverlappingFilterBank(LinearFilterBank):
    """Triangular frequency response whose vertices are along the scale

    The vertices of the filters are sampled uniformly along the passed scale.
    If the scale is nonlinear, the triangles will be asymmetrical. This is
    closely related to, but not identical to, the filters described in
    [povey2011]_ and [young]_.

    Parameters
    ----------
    scaling_function : pydrobert.speech.ScalingFunction, str, or dict
        Dictates the layout of filters in the Fourier domain. Can be
        a ScalingFunction or something compatible with
        `pydrobert.speech.alias_factory_subclass_from_arg`
    num_filts : int, optional
        The number of filters in the bank
    high_hz, low_hz : float, optional
        The topmost and bottommost edge of the filters, respectively.
        The default for high_hz is the Nyquist
    sampling_rate : float, optional
        The sampling rate (cycles/sec) of the target recordings
    analytic : bool, optional
        Whether to use an analytic form of the bank. The analytic form is
        easily derived from the real form in [povey2011]_ and [young]_. Since
        the filter is compactly supported in frequency, the analytic form is
        simply the suppression of the ``[-pi, 0)`` frequencies

    Attributes
    ----------
    centers_hz : tuple
    is_real : bool
    is_analytic : bool
    num_filts : int
    sampling_rate : float
    supports_hz : tuple
    supports : tuple
    supports_ms : tuple

    Raises
    ------
    ValueError
        If `high_hz` is above the Nyquist, or `low_hz` is below 0, or
        ``high_hz <= low_hz``
    """

    aliases = {'tri', 'triangular'}

    def __init__(
            self, scaling_function, num_filts=40, high_hz=None, low_hz=20.,
            sampling_rate=16000, analytic=False):
        scaling_function = alias_factory_subclass_from_arg(
            ScalingFunction, scaling_function)
        if low_hz < 0 or (
                high_hz and (
                    high_hz <= low_hz or high_hz > sampling_rate // 2)):
            raise ValueError(
                'Invalid frequency range: ({:.2f},{:.2f}'.format(
                    low_hz, high_hz))
        self._rate = sampling_rate
        if high_hz is None:
            high_hz = sampling_rate // 2
        # compute vertices
        scale_low = scaling_function.hertz_to_scale(low_hz)
        scale_high = scaling_function.hertz_to_scale(high_hz)
        scale_delta = (scale_high - scale_low) / (num_filts + 1)
        self._vertices = tuple(
            scaling_function.scale_to_hertz(scale_low + scale_delta * idx)
            for idx in range(0, num_filts + 2)
        )
        self._analytic = analytic

    @property
    def is_real(self):
        return not self._analytic

    @property
    def is_analytic(self):
        return self._analytic

    @property
    def is_zero_phase(self):
        return True

    @property
    def num_filts(self):
        return len(self._vertices) - 2

    @property
    def sampling_rate(self):
        return self._rate

    @property
    def centers_hz(self):
        """The point of maximum gain in each filter's frequency response, in Hz

        This property gives the so-called "center frequencies" - the
        point of maximum gain - of each filter.
        """
        return self._vertices[1:-1]

    @property
    def supports_hz(self):
        return tuple(
            (low, high)
            for low, high in zip(self._vertices[:-2], self._vertices[2:])
        )

    @property
    def supports(self):
        # A given filter is bound from above by
        # 2(w_r - w_l) / ((w_c - w_l)(w_r - w_c)t^2pi)
        supports = []
        for idx in range(len(self._vertices) - 2):
            left = hertz_to_angular(self._vertices[idx], self._rate)
            mid = hertz_to_angular(self._vertices[idx + 1], self._rate)
            right = hertz_to_angular(self._vertices[idx + 2], self._rate)
            K = np.sqrt(8 * (right - left) / np.pi)
            K /= np.sqrt(config.EFFECTIVE_SUPPORT_THRESHOLD)
            K /= np.sqrt(mid - left) * np.sqrt(right - mid)
            K = int(np.ceil(K))
            supports.append((- K // 2 - 1, K // 2 + 1))
        return tuple(supports)

    def get_impulse_response(self, filt_idx, width):
        left = hertz_to_angular(self._vertices[filt_idx], self._rate)
        mid = hertz_to_angular(self._vertices[filt_idx + 1], self._rate)
        right = hertz_to_angular(self._vertices[filt_idx + 2], self._rate)
        res = np.zeros(
            width, dtype=np.complex128 if self._analytic else np.float64)
        # for numerical stability (angles can get pretty small)
        if right - mid > mid - left:
            denom = right - mid
            div_term = mid - left
        else:
            denom = mid - left
            div_term = right - mid
        denom *= (int(self._analytic) + 1) * np.pi
        for t in range(1, width + 1):
            if self._analytic:
                numer = (right - left) / div_term * np.exp(1j * mid * t)
                numer -= (right - mid) / div_term * np.exp(1j * left * t)
                numer -= (mid - left) / div_term * np.exp(1j * right * t)
            else:
                numer = (right - left) / div_term * np.cos(mid * t)
                numer -= (right - mid) / div_term * np.cos(left * t)
                numer -= (mid - left) / div_term * np.cos(right * t)
            val = numer / t ** 2
            if t < width:
                res[t] += val
                res[-t] += val.conj()
            else:
                res[0] += val
        numer = mid / div_term * (right ** 2 - left ** 2)
        numer += right / div_term * (left ** 2 - mid ** 2)
        numer += left / div_term * (mid ** 2 - right ** 2)
        res[0] += numer / 2
        res /= denom
        return res

    def get_frequency_response(self, filt_idx, width, half=False):
        left = self._vertices[filt_idx]
        mid = self._vertices[filt_idx + 1]
        right = self._vertices[filt_idx + 2]
        left_idx = int(np.ceil(width * left / self._rate))
        right_idx = int(width * right / self._rate)
        assert self._rate * (left_idx - 1) / width <= left
        assert self._rate * (right_idx + 1) / width >= right, width
        dft_size = width
        if half:
            if width % 2:
                dft_size = (width + 1) // 2
            else:
                dft_size = width // 2 + 1
        res = np.zeros(dft_size, dtype=np.float64)
        for idx in range(left_idx, min(dft_size, right_idx + 1)):
            hz = self._rate * idx / width
            if hz <= mid:
                val = (hz - left) / (mid - left)
            else:
                val = (right - hz) / (right - mid)
            res[idx] = val
            if not half and not self._analytic:
                res[-idx] = val
        return res

    def get_truncated_response(self, filt_idx, width):
        left = self._vertices[filt_idx]
        mid = self._vertices[filt_idx + 1]
        right = self._vertices[filt_idx + 2]
        left_idx = int(np.ceil(width * left / self._rate))
        right_idx = int(width * right / self._rate)
        assert self._rate * (left_idx - 1) / width <= left
        assert self._rate * (right_idx + 1) / width >= right, width
        res = np.zeros(1 + right_idx - left_idx, dtype=np.float64)
        for idx in range(left_idx, min(width, right_idx + 1)):
            hz = self._rate * idx / width
            if hz <= mid:
                res[idx - left_idx] = (hz - left) / (mid - left)
            else:
                res[idx - left_idx] = (right - hz) / (right - mid)
        return left_idx, res


class Fbank(LinearFilterBank):
    '''A mel-triangular filter bank that is square-rooted

    An ``Fbank`` instance is intended to replicate the filters from Kaldi
    [povey2011]_ and HTK [young]_. Its scale is fixed to Mel-scale. Like a
    ``TriangularOverlappingFilterBank``, ``Fbank`` places the vertices of
    triangular filters uniformly along the target scale. However, an ``Fbank``
    is triangular in the Mel-scale, whereas the triangular bank is triangular
    in frequency.

    Parameters
    ----------
    num_filts : int, optional
        The number of filters in the bank
    high_hz, low_hz : float, optional
        The topmost and bottommost edge of the filters, respectively.
        The default for high_hz is the Nyquist
    sampling_rate : float, optional
        The sampling rate (cycles/sec) of the target recordings
    analytic : bool, optional
        Whether to use an analytic form of the bank. The analytic form
        is easily derived from the real form in [1]_ and [2]_. Since
        the filter is compactly supported in frequency, the analytic
        form is simply the suppression of the ``[-pi, 0)`` frequencies

    Attributes
    ----------
    centers_hz : tuple
    is_real : bool
    is_analytic : bool
    num_filts : int
    sampling_rate : float
    supports_hz : tuple
    supports : tuple
    supports_ms : tuple

    Notes
    -----
    In a standard mel-filterbank spectrogram, the power spectrum is calculated
    before filtering. This module's spectrogram takes the power spectrum after
    filtering. To recreate the frequency response of the alternate order, we
    can take the pointwise square root of the frequency response.
    '''

    aliases = {'fbank'}

    def __init__(
            self, num_filts=40, high_hz=None, low_hz=20., sampling_rate=16000,
            analytic=False):
        scaling_function = MelScaling()
        if low_hz < 0 or (
                high_hz and (
                    high_hz <= low_hz or high_hz > sampling_rate // 2)):
            raise ValueError(
                'Invalid frequency range: ({:.2f},{:.2f}'.format(
                    low_hz, high_hz))
        self._rate = sampling_rate
        if high_hz is None:
            high_hz = sampling_rate // 2
        # compute vertices
        scale_low = scaling_function.hertz_to_scale(low_hz)
        scale_high = scaling_function.hertz_to_scale(high_hz)
        scale_delta = (scale_high - scale_low) / (num_filts + 1)
        self._vertices = tuple(
            scaling_function.scale_to_hertz(scale_low + scale_delta * idx)
            for idx in range(0, num_filts + 2)
        )
        self._analytic = analytic

    @property
    def is_real(self):
        return not self._analytic

    @property
    def is_analytic(self):
        return self._analytic

    @property
    def is_zero_phase(self):
        return True

    @property
    def num_filts(self):
        return len(self._vertices) - 2

    @property
    def sampling_rate(self):
        return self._rate

    @property
    def centers_hz(self):
        """The point of maximum gain in each filter's frequency response, in Hz

        This property gives the so-called "center frequencies" - the
        point of maximum gain - of each filter.
        """
        return self._vertices[1:-1]

    @property
    def supports_hz(self):
        return tuple(
            (low, high)
            for low, high in zip(self._vertices[:-2], self._vertices[2:])
        )

    @property
    def supports(self):
        # A given filter is bound above for t > 0 by
        # ((w_r - w_c) ** .5 + (w_c - w_l) ** .5) /
        #   (2 ** 3 * t ** 3 * (w_c - w_l) * (w_r - w_c) * pi) ** .5
        supports = []
        for idx in range(len(self._vertices) - 2):
            left = hertz_to_angular(self._vertices[idx], self._rate)
            mid = hertz_to_angular(self._vertices[idx + 1], self._rate)
            right = hertz_to_angular(self._vertices[idx + 2], self._rate)
            K = right - left + 2 * ((right - mid) * (mid - left)) ** 2
            K /= config.EFFECTIVE_SUPPORT_THRESHOLD ** 2 * np.pi
            K /= (right - mid) * (mid - left)
            K /= np.sqrt(config.EFFECTIVE_SUPPORT_THRESHOLD)
            K /= np.sqrt(mid - left) * np.sqrt(right - mid)
            K **= .3333
            K = int(np.ceil(K))
            supports.append((- K // 2 - 1, K // 2 + 1))
        return tuple(supports)

    def get_impulse_response(self, filt_idx, width):
        # For the time being, I'll just invert the frequency response
        if self.is_analytic:
            freq_response = self.get_frequency_response(
                filt_idx, width, half=False)
            return np.fft.ifft(freq_response)
        else:
            freq_response = self.get_frequency_response(
                filt_idx, width, half=True)
            return np.fft.irfft(freq_response, n=width)

    def get_frequency_response(self, filt_idx, width, half=False):
        scaling_function = MelScaling()
        left_hz = self._vertices[filt_idx]
        mid_hz = self._vertices[filt_idx + 1]
        right_hz = self._vertices[filt_idx + 2]
        left_mel = scaling_function.hertz_to_scale(left_hz)
        mid_mel = scaling_function.hertz_to_scale(mid_hz)
        right_mel = scaling_function.hertz_to_scale(right_hz)
        left_idx = int(np.ceil(width * left_hz / self._rate))
        right_idx = int(width * right_hz / self._rate)
        assert self._rate * (left_idx - 1) / width <= left_hz
        assert self._rate * (right_idx + 1) / width >= right_hz, width
        dft_size = width
        if half:
            if width % 2:
                dft_size = (width + 1) // 2
            else:
                dft_size = width // 2 + 1
        res = np.zeros(dft_size, dtype=np.float64)
        for idx in range(left_idx, min(dft_size, right_idx + 1)):
            hz = self._rate * idx / width
            mel = scaling_function.hertz_to_scale(hz)
            if mel <= mid_mel:
                val = (mel - left_mel) / (mid_mel - left_mel)
            else:
                val = (right_mel - mel) / (right_mel - mid_mel)
            res[idx] = val ** .5
            if not half and not self._analytic:
                res[-idx] = val ** .5
        return res

    def get_truncated_response(self, filt_idx, width):
        scaling_function = MelScaling()
        left_hz = self._vertices[filt_idx]
        mid_hz = self._vertices[filt_idx + 1]
        right_hz = self._vertices[filt_idx + 2]
        left_mel = scaling_function.hertz_to_scale(left_hz)
        mid_mel = scaling_function.hertz_to_scale(mid_hz)
        right_mel = scaling_function.hertz_to_scale(right_hz)
        left_idx = int(np.ceil(width * left_hz / self._rate))
        right_idx = int(width * right_hz / self._rate)
        assert self._rate * (left_idx - 1) / width <= left_hz
        assert self._rate * (right_idx + 1) / width >= right_hz, width
        res = np.zeros(min(width, right_idx + 1) - left_idx, dtype=np.float64)
        for idx in range(left_idx, min(width, right_idx + 1)):
            hz = self._rate * idx / width
            mel = scaling_function.hertz_to_scale(hz)
            if mel <= mid_mel:
                res[idx - left_idx] = (mel - left_mel) / (mid_mel - left_mel)
            else:
                res[idx - left_idx] = (right_mel - mel) / (right_mel - mid_mel)
        return left_idx, res ** .5


class GaborFilterBank(LinearFilterBank):
    r"""Gabor filters with ERBs between points from a scale

    Gabor filters are complex, mostly analytic filters that have a Gaussian
    envelope in both the time and frequency domains. They are defined as

    .. math::

         f(t) = C \sigma^{-1/2} \pi^{-1/4}
                e^{\frac{-t^2}{2\sigma^2} + i\xi t}

    in the time domain and

    .. math::

         \widehat{f}(\omega) = C \sqrt{2\sigma} \pi^{1/4}
                               e^{\frac{-\sigma^2(\xi - \omega)^2}{2}}

    in the frequency domain. Though Gaussians never truly reach 0, in either
    domain, they are effectively compactly supported. Gabor filters are optimal
    with respect to their time-bandwidth product.

    `scaling_function` is used to split up the frequencies between `high_hz`
    and `low_hz` into a series of filters. Every subsequent filter's width is
    scaled such that, if the filters are all of the same height, the
    intersection with the precedent filter's response matches the filter's
    Equivalent Rectangular Bandwidth (``erb == True``) or its 3dB bandwidths (
    ``erb == False``). The ERB is the width of a rectangular filter with the
    same height as the filter's maximum frequency response that has the same
    :math:`L_2` norm.

    Parameters
    ----------
    scaling_function : pydrobert.speech.ScalingFunction, str, or dict
        Dictates the layout of filters in the Fourier domain. Can be
        a ScalingFunction or something compatible with
        `pydrobert.speech.alias_factory_subclass_from_arg`
    num_filts : int
        The number of filters in the bank
    high_hz, low_hz : float, optional
        The topmost and bottommost edge of the filters, respectively.
        The default for high_hz is the Nyquist
    sampling_rate : float, optional
        The sampling rate (cycles/sec) of the target recordings
    scale_l2_norm : bool
        Whether to scale the l2 norm of each filter to 1. Otherwise the
        frequency response of each filter will max out at an absolute
        value of 1.
    erb : bool

    Attributes
    ----------
    centers_hz : tuple
    is_real : bool
    is_analytic : bool
    num_filts : int
    sampling_rate : float
    supports_hz : tuple
    supports : tuple
    supports_ms : tuple
    scaled_l2_norm : bool
    erb : bool

    See Also
    --------
    pydrobert.speech.config.EFFECTIVE_SUPPORT_THRESHOLD : the absolute
        value below which counts as zero
    """

    aliases = {'gabor'}

    def __init__(
            self, scaling_function, num_filts=40, high_hz=None, low_hz=20.,
            sampling_rate=16000, scale_l2_norm=False, erb=False):
        scaling_function = alias_factory_subclass_from_arg(
            ScalingFunction, scaling_function)
        self._scale_l2_norm = scale_l2_norm
        self._erb = erb
        if low_hz < 0 or (
                high_hz and (
                    high_hz <= low_hz or high_hz > sampling_rate // 2)):
            raise ValueError(
                'Invalid frequency range: ({:.2f},{:.2f}'.format(
                    low_hz, high_hz))
        self._rate = sampling_rate
        if high_hz is None:
            high_hz = sampling_rate // 2
        scale_low = scaling_function.hertz_to_scale(low_hz)
        scale_high = scaling_function.hertz_to_scale(high_hz)
        scale_delta = (scale_high - scale_low) / (num_filts + 1)
        # edges dictate the points where filters should intersect. We
        # make a pretend intersection halfway between low_hz and
        # the first filter center in the scaled domain. Likewise with
        # high_hz and the last filter center. Intersections are spaced
        # uniformly in the scaled domain
        edges = tuple(
            scaling_function.scale_to_hertz(
                scale_low + scale_delta * (idx + .5))
            for idx in range(0, num_filts + 1)
        )
        centers_hz = []
        centers_ang = []
        stds = []
        supports_ang = []
        supports = []
        wrap_supports_ang = []
        self._wrap_below = False
        log_2 = np.log(2)
        log_pi = np.log(np.pi)
        t_support_const = -2 * np.log(config.EFFECTIVE_SUPPORT_THRESHOLD)
        f_support_const = t_support_const
        if scale_l2_norm:
            f_support_const += log_2 + .5 * log_pi
            t_support_const -= .5 * log_pi
        else:
            t_support_const -= log_2 + log_pi
        if erb:
            bandwidth_const = np.sqrt(np.pi) / 2
        else:
            bandwidth_const = np.sqrt(3 / 10 * np.log(10))
        for left_intersect, right_intersect in zip(edges[:-1], edges[1:]):
            center_hz = (left_intersect + right_intersect) / 2
            center_ang = hertz_to_angular(center_hz, self._rate)
            std = bandwidth_const / hertz_to_angular(
                center_hz - left_intersect, self._rate)
            log_std = np.log(std)
            if scale_l2_norm:
                diff_ang = np.sqrt(log_std + f_support_const) / std
                wrap_diff_ang = np.sqrt(
                    log_std + f_support_const + log_2) / std
                diff_samps = int(np.ceil(
                    std * np.sqrt(t_support_const - log_std)))
            else:
                diff_ang = np.sqrt(f_support_const) / std
                wrap_diff_ang = np.sqrt(f_support_const + log_2) / std
                diff_samps = int(np.ceil(std * np.sqrt(
                    t_support_const - 2 * log_std)))
            supp_ang_low = center_ang - diff_ang
            if supp_ang_low < 0:
                self._wrap_below = True
            centers_hz.append(center_hz)
            centers_ang.append(center_ang)
            supports_ang.append((center_ang - diff_ang, center_ang + diff_ang))
            wrap_supports_ang.append(2 * wrap_diff_ang)
            supports.append((-diff_samps, diff_samps))
            stds.append(std)
        self._centers_ang = tuple(centers_ang)
        self._centers_hz = tuple(centers_hz)
        self._stds = tuple(stds)
        self._supports_ang = tuple(supports_ang)
        self._wrap_supports_ang = tuple(wrap_supports_ang)
        self._supports_hz = tuple(
            (
                angular_to_hertz(ang_l, self._rate),
                angular_to_hertz(ang_h, self._rate),
            )
            for ang_l, ang_h in supports_ang
        )
        self._supports = tuple(supports)

    @property
    def is_real(self):
        return False

    @property
    def is_analytic(self):
        return not self._wrap_below

    @property
    def num_filts(self):
        return len(self._centers_hz)

    @property
    def is_zero_phase(self):
        return True

    @property
    def sampling_rate(self):
        return self._rate

    @property
    def centers_hz(self):
        """The point of maximum gain in each filter's frequency response, in Hz

        This property gives the so-called "center frequencies" - the
        point of maximum gain - of each filter.
        """
        return self._centers_hz

    @property
    def supports_hz(self):
        return self._supports_hz

    @property
    def supports(self):
        return self._supports

    @property
    def scaled_l2_norm(self):
        return self._scale_l2_norm

    @property
    def erb(self):
        return self._erb

    def get_impulse_response(self, filt_idx, width):
        center_ang = self._centers_ang[filt_idx]
        std = self._stds[filt_idx]
        res = np.zeros(width, dtype=np.complex128)
        if self._scale_l2_norm:
            const_term = -.5 * np.log(std) - .25 * np.log(np.pi)
        else:
            const_term = -.5 * np.log(2 * np.pi) - np.log(std)
        denom_term = 2 * std ** 2
        for t in range(width + 1):
            val = -t ** 2 / denom_term + const_term + 1j * center_ang * t
            val = np.exp(val)
            if t != width:
                res[t] += val
            if t:
                res[-t] += val.conj()
        return res

    def get_frequency_response(self, filt_idx, width, half=False):
        center_ang = self._centers_ang[filt_idx]
        lowest_ang, highest_ang = self._supports_ang[filt_idx]
        std = self._stds[filt_idx]
        dft_size = width
        if half:
            if width % 2:
                dft_size = (width + 1) // 2
            else:
                dft_size = width // 2 + 1
        res = np.zeros(dft_size, dtype=np.float64)
        if self._scale_l2_norm:
            const_term = .5 * np.log(2 * std) + .25 * np.log(np.pi)
        else:
            const_term = 0
        num_term = -(std ** 2) / 2
        for idx in range(dft_size):
            for period in range(
                    -1 - int(max(-lowest_ang, 0) / (2 * np.pi)),
                    2 + int(highest_ang / (2 * np.pi))):
                omega = (idx / width + period) * 2 * np.pi
                val = num_term * (center_ang - omega) ** 2 + const_term
                val = np.exp(val)
                res[idx] += val
        return res

    def get_truncated_response(self, filt_idx, width):
        # wrap_supports_ang contains the angular supports of each filter
        # if the effective support threshold were halved. If this
        # support exceeds the 2pi period, overlap from aliasing in the
        # periphery will exceed the effective support, meaning the
        # entire period lies in the support
        if self._wrap_supports_ang[filt_idx] >= 2 * np.pi:
            return 0, self.get_frequency_response(filt_idx, width)
        center_ang = self._centers_ang[filt_idx]
        std = self._stds[filt_idx]
        lowest_ang, highest_ang = self._supports_ang[filt_idx]
        left_idx = int(np.ceil(width * lowest_ang / (2 * np.pi)))
        right_idx = int(width * highest_ang / (2 * np.pi))
        res = np.zeros(1 + right_idx - left_idx, dtype=np.float64)
        if self._scale_l2_norm:
            const_term = .5 * np.log(2 * std) + .25 * np.log(np.pi)
        else:
            const_term = 0
        num_term = -(std ** 2) / 2
        for idx in range(left_idx, right_idx + 1):
            for period in range(
                    -int(max(-lowest_ang, 0) / (2 * np.pi)),
                    1 + int(highest_ang / (2 * np.pi))):
                omega = (idx / width + period) * 2 * np.pi
                val = num_term * (center_ang - omega) ** 2 + const_term
                val = np.exp(val)
                res[idx - left_idx] += val
        return left_idx % width, res


class ComplexGammatoneFilterBank(LinearFilterBank):
    r'''Gammatone filters with complex carriers

    A complex gammatone filter [flanagan1960]_ [aertsen1981]_ can be defined as

    .. math::

        h(t) = c t^{n - 1} e^{- \alpha t + i\xi t} u(t)

    in the time domain, where :math:`\alpha` is the bandwidth parameter,
    :math:`\xi` is the carrier frequency, :math:`n` is the order of the
    function, :math:`u(t)` is the step function, and :math:`c` is a
    normalization constant. In the frequency domain, the filter is defined as

    .. math::

        H(\omega) = \frac{c(n - 1)!)}{\left(
            \alpha + i(\omega - \xi) \right)^n}

    For large :math:`\xi`, the complex gammatone is approximately
    analytic.

    `scaling_function` is used to split up the frequencies between `high_hz`
    and `low_hz` into a series of filters. Every subsequent filter's width is
    scaled such that, if the filters are all of the same height, the
    intersection with the precedent filter's response matches the filter's
    Equivalent Rectangular Bandwidth (``erb == True``) or its 3dB bandwidths
    (``erb == False``). The ERB is the width of a rectangular filter with the
    same height as the filter's maximum frequency response that has the same
    :math:`L_2` norm.

    Parameters
    ----------
    scaling_function : pydrobert.speech.ScalingFunction, str, or dict
        Dictates the layout of filters in the Fourier domain. Can be
        a ScalingFunction or something compatible with
        `pydrobert.speech.alias_factory_subclass_from_arg`
    num_filts : int, optional
        The number of filters in the bank
    high_hz, low_hz : float, optional
        The topmost and bottommost edge of the filters, respectively.
        The default for high_hz is the Nyquist
    sampling_rate : float, optional
        The sampling rate (cycles/sec) of the target recordings
    order : int, optional
        The :math:`n` parameter in the Gammatone. Should be positive.
        Larger orders will make the gammatone more symmetrical.
    max_centered : bool, optional
        While normally causal, setting `max_centered` to true will shift
        all filters in the bank such that the maximum absolute value
        in time is centered at sample 0.
    scale_l2_norm : bool
        Whether to scale the l2 norm of each filter to 1. Otherwise the
        frequency response of each filter will max out at an absolute
        value of 1.
    erb : bool

    Attributes
    ----------
    centers_hz : tuple
    is_real : bool
    is_analytic : bool
    num_filts : int
    order : int
    sampling_rate : float
    supports_hz : tuple
    supports : tuple
    supports_ms : tuple
    scaled_l2_norm : bool
    erb : bool

    See Also
    --------
    pydrobert.speech.config.EFFECTIVE_SUPPORT_THRESHOLD : the absolute
        value below which counts as zero
    '''

    aliases = {'gammatone', 'tonebank'}

    def __init__(
            self, scaling_function, num_filts=40, high_hz=None, low_hz=20.,
            sampling_rate=16000, order=4, max_centered=False,
            scale_l2_norm=False, erb=False):
        scaling_function = alias_factory_subclass_from_arg(
            ScalingFunction, scaling_function)
        self._scale_l2_norm = scale_l2_norm
        self._erb = erb
        if low_hz < 0 or (
                high_hz and (
                    high_hz <= low_hz or high_hz > sampling_rate // 2)):
            raise ValueError(
                'Invalid frequency range: ({:.2f},{:.2f}'.format(
                    low_hz, high_hz))
        if not isinstance(order, int) or order <= 0:
            raise ValueError('order must be a positive integer')
        self._order = order
        self._rate = sampling_rate
        if high_hz is None:
            high_hz = sampling_rate // 2
        scale_low = scaling_function.hertz_to_scale(low_hz)
        scale_high = scaling_function.hertz_to_scale(high_hz)
        scale_delta = (scale_high - scale_low) / (num_filts + 1)
        # see gabor filters for more info
        edges = tuple(
            scaling_function.scale_to_hertz(
                scale_low + scale_delta * (idx + .5))
            for idx in range(0, num_filts + 1)
        )
        self._centers_hz = []
        self._xis = []
        self._alphas = []
        self._cs = []
        self._offsets = []
        self._supports = []
        self._supports_ang = []
        self._wrap_supports_ang = []
        self._wrap_below = False
        log_eps = np.log(config.EFFECTIVE_SUPPORT_THRESHOLD)
        log_double_factorial = np.log(np.math.factorial(2 * order - 2))
        log_factorial = np.log(np.math.factorial(order - 1))
        log_2 = np.log(2)
        if erb:
            alpha_const = (order - 1) * log_2
            alpha_const += 2 * log_factorial
            alpha_const -= log_double_factorial
        else:
            alpha_const = - .5 * np.log(4 * (2 ** (1 / order)) - 4)
        for left_intersect, right_intersect in zip(edges[:-1], edges[1:]):
            center_hz = (left_intersect + right_intersect) / 2
            xi = hertz_to_angular(center_hz, self._rate)
            log_alpha = alpha_const + np.log(hertz_to_angular(
                right_intersect - left_intersect, self._rate))
            alpha = np.exp(log_alpha)
            if scale_l2_norm:
                log_c = .5 * (
                    log_2 + log_alpha + log_double_factorial)
                log_c -= order * (log_alpha + log_2)
            else:
                log_c = order * log_alpha - log_factorial
            c = np.exp(log_c)
            if max_centered:
                offset = -(order - 1) / alpha
            else:
                offset = 0
            supp_a = (2 / order) * (log_c + log_factorial - log_eps)
            wrap_supp_a = supp_a + (2 / order) * log_2
            supp_b = np.exp(2 * log_alpha)
            diff_ang = (np.exp(supp_a) - supp_b) ** .5
            wrap_diff_ang = (np.exp(wrap_supp_a) - supp_b) ** .5
            self._centers_hz.append(center_hz)
            self._xis.append(xi)
            self._alphas.append(alpha)
            self._cs.append(c)
            self._offsets.append(offset)
            self._supports.append(self._calculate_temp_support(-1))
            self._supports_ang.append((xi - diff_ang, xi + diff_ang))
            if self._supports_ang[-1][0] < 0:
                self._wrap_below = True
            self._wrap_supports_ang.append(2 * wrap_diff_ang)
        self._xis = tuple(self._xis)
        self._cs = tuple(self._cs)
        self._alphas = tuple(self._alphas)
        self._offsets = tuple(self._offsets)
        self._centers_hz = tuple(self._centers_hz)
        self._supports_ang = tuple(self._supports_ang)
        self._wrap_supports_ang = tuple(self._wrap_supports_ang)
        self._supports_hz = tuple(
            (
                angular_to_hertz(ang_l, self._rate),
                angular_to_hertz(ang_h, self._rate),
            )
            for ang_l, ang_h in self._supports_ang)
        self._supports = tuple(self._supports)

    @property
    def is_real(self):
        return False

    @property
    def is_analytic(self):
        return not self._wrap_below

    @property
    def num_filts(self):
        return len(self._centers_hz)

    @property
    def order(self):
        return self._order

    @property
    def is_zero_phase(self):
        return False

    @property
    def sampling_rate(self):
        return self._rate

    @property
    def centers_hz(self):
        """The point of maximum gain in each filter's frequency response, in Hz

        This property gives the so-called "center frequencies" - the
        point of maximum gain - of each filter.
        """
        return self._centers_hz

    @property
    def supports_hz(self):
        return self._supports_hz

    @property
    def supports(self):
        return self._supports

    @property
    def scaled_l2_norm(self):
        return self._scale_l2_norm

    @property
    def erb(self):
        return self._erb

    def get_impulse_response(self, filt_idx, width):
        left_sup, right_sup = self.supports[filt_idx]
        left_period = int(np.floor(left_sup / width))
        right_period = int(np.ceil(right_sup / width))
        res = np.zeros(width, dtype=np.complex128)
        for period in range(left_period, right_period + 1):
            for idx in range(width):
                t = period * width + idx
                res[idx] += self._h(t, filt_idx)
        return res

    def get_frequency_response(self, filt_idx, width, half=False):
        left_sup, right_sup = self._supports_ang[filt_idx]
        left_period = int(np.floor(left_sup / 2 / np.pi))
        right_period = int(np.ceil(right_sup / 2 / np.pi))
        if half:
            if width % 2:
                dft_size = (width + 1) // 2
            else:
                dft_size = width // 2 + 1
        else:
            dft_size = width
        res = np.zeros(dft_size, dtype=np.complex128)
        for period in range(left_period, right_period + 1):
            for idx in range(dft_size):
                omega = (idx / width + period) * 2 * np.pi
                res[idx] += self._H(omega, filt_idx)
        return res

    def get_truncated_response(self, filt_idx, width):
        left_sup, right_sup = self._supports_ang[filt_idx]
        wrap_ang = self._wrap_supports_ang[filt_idx]
        # wrap_ang is the additional support needed to hit
        # half the effective support threshold. If that support is
        # greater than the 2pi periodization, some points could exceed
        # the threshold due to wrapping.
        if right_sup - left_sup + wrap_ang >= 2 * np.pi:
            return 0, self.get_frequency_response(filt_idx, width)
        left_idx = int(np.ceil(width * left_sup / 2 / np.pi))
        right_idx = int(width * right_sup / 2 / np.pi)
        omega = np.arange(left_idx, right_idx + 1, dtype=np.float64)
        omega *= 2 * np.pi / width
        return left_idx % width, self._H(omega, filt_idx)

    def _h(self, t, idx):
        # calculate impulse response of filt idx at sample t
        offset = self._offsets[idx]
        if t <= offset:
            return 0j
        alpha = self._alphas[idx]
        log_c = np.log(self._cs[idx])
        xi = self._xis[idx]
        n = self._order
        r = log_c + (n - 1) * np.log(t - offset)
        r += (-alpha + 1j * xi) * (t - offset)
        return np.exp(r)

    def _H(self, omega, idx):
        # calculate frequency response of filt idx at ang freqs omega
        alpha = self._alphas[idx]
        c = self._cs[idx]
        xi = self._xis[idx]
        offset = self._offsets[idx]
        n = self._order
        numer = np.exp(-1j * omega * offset) * c * np.math.factorial(n - 1)
        denom = (alpha + 1j * (omega - xi)) ** n
        return numer / denom

    def _calculate_temp_support(self, idx):
        # calculate the nonzero region of the temp support of filt idx
        alpha = self._alphas[idx]
        c = self._cs[idx]
        xi = self._xis[idx]
        offset = self._offsets[idx]
        n = self._order
        eps = config.EFFECTIVE_SUPPORT_THRESHOLD
        if n == 1:
            right = int(np.ceil((np.log(c) - np.log(eps) / alpha)))
        else:
            def _d(t):
                # derivative of abs func
                v = c * np.exp(-alpha * t) * t ** (n - 2)
                v *= (n - 1) - alpha * t
                return v
            right = (n - 1 + np.sqrt((n - 1) / 2)) / alpha
            h_0 = np.abs(self._h(right, idx))
            while h_0 > eps:
                d_0 = _d(right)
                right -= h_0 / d_0
                h_0 = np.abs(self._h(right, idx))
        return (int(np.floor(offset)), int(np.ceil(right) + offset))


# windows


class WindowFunction(AliasedFactory):
    '''A real linear filter, usually lowpass'''

    @abc.abstractmethod
    def get_impulse_response(self, width):
        '''Write the filter into a numpy array of fixed width'''
        pass


class BartlettWindow(WindowFunction):
    '''A unit-normalized triangular window

    See Also
    --------
    numpy.bartlett
    '''

    aliases = {'bartlett', 'triangular', 'tri'}

    def get_impulse_response(self, width):
        window = np.bartlett(width)
        window /= max(1, width - 1) / 2
        return window


class BlackmanWindow(WindowFunction):
    '''A unit-normalized Blackman window

    See Also
    --------
    numpy.blackman
    '''

    aliases = {'blackman', 'black'}

    def get_impulse_response(self, width):
        window = np.blackman(width)
        window /= 0.42 * max(1, width - 1)
        return window


class HammingWindow(WindowFunction):
    '''A unit-normalized Hamming window

    See Also
    --------
    numpy.hamming
    '''

    aliases = {'hamming'}

    def get_impulse_response(self, width):
        window = np.hamming(width)
        window /= 0.54 * max(1, width - 1)
        return window


class HannWindow(WindowFunction):
    '''A unit-normalized Hann window

    See Also
    --------
    numpy.hanning
    '''

    aliases = {'hanning', 'hann'}

    def get_impulse_response(self, width):
        window = np.hanning(width)
        window /= 0.5 * max(1, width - 1)
        return window


class GammaWindow(WindowFunction):
    r'''A lowpass filter based on the Gamma function

    A Gamma function is defined as:

    .. math:: p(t; \alpha, n) = t^{n - 1} e^{-\alpha t} u(t)

    Where :math:`n` is the order of the function, :math:`\alpha`
    controls the bandwidth of the filter, and :math:`u` is the step
    function.

    This function returns a window based off a reflected Gamma function.
    :math:`\alpha` is chosen such that the maximum value of the window
    aligns with `peak`. The window is clipped to the width. For
    reasonable values of `peak` (i.e. in the last quarter of samples),
    the majority of the support should lie in this interval anyways.

    Arguments
    ---------
    order : int
    peak : float
        ``peak * width``, where ``width`` is the length of the window
        in samples, is where the approximate maximal value of the window
        lies

    Attributes
    ----------
    order : int
    peak : int
    '''

    aliases = {'gamma'}

    def __init__(self, order=4, peak=.75):
        self.order = order
        self.peak = peak

    def get_impulse_response(self, width):
        if width <= 0:
            return np.array([], dtype=float)
        elif width == 1:
            return np.array([1], dtype=float)
        peak = self.peak * width
        ret = np.arange(width - 1, -1, -1, dtype=float)
        if self.order > 1:
            alpha = (self.order - 1) / (width - peak)
            offs = width - 1
        else:
            # align alpha roughly with a support in M
            alpha = 5 / width
            offs = width
        ln_c = self.order * np.log(alpha)
        ln_c -= np.log(np.math.factorial(self.order - 1))
        ret[:offs] = ret[:offs] ** (self.order - 1) * np.exp(
            -alpha * ret[:offs] + ln_c)
        return ret
