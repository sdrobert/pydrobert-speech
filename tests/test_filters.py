from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from pydrobert.speech import filters
from pydrobert.speech.config import EFFECTIVE_SUPPORT_THRESHOLD
from pydrobert.speech.filters import GammaWindow


@pytest.fixture(params=[
    1,
    11,
], ids=[
    '1 filt',
    '11 filts',
], scope='module',)
def num_filts(request):
    return request.param


@pytest.fixture(params=[
    lambda num_filts: filters.TriangularOverlappingFilterBank(
        'mel',
        low_hz=5,
        num_filts=num_filts,
        sampling_rate=8000,
        analytic=True,
    ),
    lambda num_filts: filters.TriangularOverlappingFilterBank(
        'mel',
        low_hz=0,
        num_filts=num_filts,
        sampling_rate=8000,
        analytic=False,
    ),
    lambda num_filts: filters.Fbank(
        'mel',
        low_hz=0,
        num_filts=num_filts,
        sampling_rate=8000,
        analytic=True,
    ),
    lambda num_filts: filters.Fbank(
        'mel',
        low_hz=0,
        num_filts=num_filts,
        sampling_rate=8000,
        analytic=False,
    ),
    lambda num_filts: filters.GaborFilterBank(
        'mel',
        low_hz=0,
        num_filts=num_filts,
        sampling_rate=8000,
        erb=True,
    ),
    lambda num_filts: filters.GaborFilterBank(
        'mel',
        low_hz=0,
        num_filts=num_filts,
        sampling_rate=8000,
        erb=False,
    ),
    lambda num_filts: filters.ComplexGammatoneFilterBank(
        'mel',
        low_hz=0,
        num_filts=num_filts,
        sampling_rate=8000,
        max_centered=True,
    ),
], ids=[
    'triangular_analytic',
    'triangular',
    'fbank_analytic',
    'fbank',
    'gabor_erb',
    'gabor',
    'gammatone',
])
def bank(request, num_filts):
    return request.param(num_filts)


def test_truncated_matches_full(bank):
    for filt_idx in range(bank.num_filts):
        left_hz, right_hz = bank.supports_hz[filt_idx]
        left_samp, right_samp = bank.supports[filt_idx]
        dft_size = int(max(
            (right_samp - left_samp) * (1 + np.random.random()),
            2 * bank.sampling_rate / (right_hz - left_hz),
            1,
        ))
        left_period = int(np.floor(left_hz / bank.sampling_rate))
        right_period = int(np.ceil(right_hz / bank.sampling_rate))
        full_response = bank.get_frequency_response(filt_idx, dft_size)
        bin_idx, truncated = bank.get_truncated_response(filt_idx, dft_size)
        challenge = np.zeros(dft_size, dtype=truncated.dtype)
        wrap = min(bin_idx + len(truncated), dft_size) - bin_idx
        challenge[bin_idx:bin_idx + wrap] = truncated[:wrap]
        challenge[:len(truncated) - wrap] = truncated[wrap:]
        if bank.is_real:
            challenge[
                len(challenge) - bin_idx - len(truncated) + 1:
                len(challenge) - bin_idx + 1
            ] = truncated[:None if bin_idx else 0:-1].conj()
        bad_idx = np.where(np.logical_not(np.isclose(
            full_response, challenge, atol=EFFECTIVE_SUPPORT_THRESHOLD)))
        assert np.allclose(
            full_response, challenge,
            atol=(right_period - left_period) * EFFECTIVE_SUPPORT_THRESHOLD
        ), 'idx: {} threshold:{} full:{} challenge:{}'.format(
            filt_idx, EFFECTIVE_SUPPORT_THRESHOLD,
            full_response[bad_idx], challenge[bad_idx]
        )


def test_frequency_matches_impulse(bank):
    for filt_idx in range(bank.num_filts):
        left_hz, right_hz = bank.supports_hz[filt_idx]
        left_samp, right_samp = bank.supports[filt_idx]
        required_freq_size = 2 * bank.sampling_rate / (right_hz - left_hz)
        required_temp_size = right_samp - left_samp
        if required_temp_size < 5 or required_freq_size < 5:
            # FIXME(sdrobert): this is a stopgap for when filters are
            # *too* localized in time or frequency. This'll cause too
            # much attenuation/gain in one domain or the other.
            continue
        dft_size = int(max(
            # allow over- or under-sampling
            (right_samp - left_samp),
            2 * bank.sampling_rate / (right_hz - left_hz)
        ))
        X = bank.get_frequency_response(filt_idx, dft_size)
        x = bank.get_impulse_response(filt_idx, dft_size)
        # the absolute tolerance is so high because spectral leakage
        # will muck with the isometry. .001 is enough to say we're in
        # the right ballpark... probably
        assert np.allclose(np.fft.ifft(X), x, atol=1e-3), (len(x), filt_idx)


def test_half_response_matches_full(bank):
    for filt_idx in range(bank.num_filts):
        dft_size = bank.supports[filt_idx][1] - bank.supports[filt_idx][0]
        Xh = bank.get_frequency_response(filt_idx, dft_size, half=True)
        X = bank.get_frequency_response(filt_idx, dft_size, half=False)
        assert np.allclose(X[:len(Xh)], Xh)


def test_zero_outside_freq_support(bank):
    for filt_idx in range(bank.num_filts):
        left_hz, right_hz = bank.supports_hz[filt_idx]
        dft_size = int(max(1, 2 * bank.sampling_rate / (right_hz - left_hz)))
        left_period = int(np.floor(left_hz / bank.sampling_rate))
        right_period = int(np.ceil(right_hz / bank.sampling_rate))
        if right_period - left_period > 2:
            continue
        zero_mask = np.ones(dft_size, dtype=bool)
        for period in range(left_period, right_period + 1):
            for idx in range(dft_size):
                freq = (idx / dft_size + period) * bank.sampling_rate
                zero_mask[idx] &= (freq < left_hz) | (freq > right_hz)
        if bank.is_real:
            zero_mask[1:] &= zero_mask[-1:0:-1]
        if not np.any(zero_mask):
            continue
        X = bank.get_frequency_response(filt_idx, dft_size)
        assert np.allclose(
            X[zero_mask], 0,
            atol=(right_period - left_period) * EFFECTIVE_SUPPORT_THRESHOLD
        )


def test_zero_outside_temp_support(bank):
    for filt_idx in range(bank.num_filts):
        left_samp, right_samp = bank.supports[filt_idx]
        width = int(max(1, right_samp - left_samp))
        left_period = int(np.floor(left_samp / width))
        right_period = int(np.ceil(right_samp / width))
        if right_period - left_period > 2:
            continue
        zero_mask = np.ones(width, dtype=bool)
        for period in range(left_period, right_period + 1):
            for idx in range(width):
                t = idx + period * width
                zero_mask[idx] &= (t < left_samp) | (t > right_samp)
        if not np.any(zero_mask):
            continue
        x = bank.get_impulse_response(filt_idx, width)
        assert np.allclose(
            x[zero_mask], 0,
            atol=(right_period - left_period) * EFFECTIVE_SUPPORT_THRESHOLD
        )


@pytest.mark.parametrize('window_size', [10, 100, 1000])
@pytest.mark.parametrize('peak_ratio', [.5, .75, .9])
@pytest.mark.parametrize('order', [2, 4])
def test_gamma_window_peak_matches(window_size, peak_ratio, order):
    expected_max_idx = window_size * peak_ratio
    window = GammaWindow(order=order, peak=peak_ratio).get_impulse_response(
        window_size)
    # if expected_max_idx is a fraction, we don't know what side the peak will
    # fall on a priori because the gamma function does not have symmetric
    # curvature
    max_idx = np.argmax(window)
    assert int(expected_max_idx) == max_idx or \
        int(expected_max_idx) == max_idx + 1
