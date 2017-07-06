# pylint: skip-file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pydrobert.signal import EFFECTIVE_SUPPORT_THRESHOLD
from pydrobert.signal.util import hertz_to_angular

def test_truncated_matches_full(bank):
    for filt_idx in range(bank.num_filts):
        dft_size = int(
            bank.supports[filt_idx] * (np.abs(np.random.random()) + 1))
        full_response = bank.get_frequency_response(filt_idx, dft_size)
        bin_idx, truncated = bank.get_truncated_response(
            filt_idx, dft_size)
        challenge = np.zeros(dft_size, dtype=truncated.dtype)
        wrap = min(bin_idx + len(truncated), dft_size) - bin_idx
        challenge[bin_idx:bin_idx + wrap] = truncated[:wrap]
        challenge[:len(truncated) - wrap] = truncated[wrap:]
        if bank.is_real:
            challenge[
                len(challenge) - bin_idx - len(truncated) + 1:
                len(challenge) - bin_idx + 1] += truncated[::-1].conj()
        bad_idx = np.where(np.logical_not(np.isclose(
            full_response, challenge, atol=EFFECTIVE_SUPPORT_THRESHOLD)))
        assert np.allclose(
            full_response, challenge,
            atol=EFFECTIVE_SUPPORT_THRESHOLD
        ), 'idx: {} threshold:{} full:{} challenge:{}'.format(
            filt_idx, EFFECTIVE_SUPPORT_THRESHOLD,
            full_response[bad_idx], challenge[bad_idx]
        )

def test_frequency_matches_impulse(bank):
    for filt_idx in range(bank.num_filts):
        left_hz, right_hz = bank.supports_hz[filt_idx]
        dft_size = int(max(
            # allow over- or under-sampling
            bank.supports[filt_idx] * (1 + np.random.random()),
            2 * bank.sampling_rate / (right_hz - left_hz),
        ))
        X = bank.get_frequency_response(filt_idx, dft_size)
        x = bank.get_impulse_response(filt_idx, dft_size)
        # the absolute tolerance is so high because spectral leakage
        # will muck with the isometry. .001 is enough to say we're in
        # the right ballpark... probably
        assert np.allclose(np.fft.ifft(X), x, atol=1e-3), len(x)

def test_half_response_matches_full(bank):
    for filt_idx in range(bank.num_filts):
        dft_size = bank.supports[filt_idx]
        Xh = bank.get_frequency_response(filt_idx, dft_size, half=True)
        X = bank.get_frequency_response(filt_idx, dft_size, half=False)
        assert np.allclose(X[:len(Xh)], Xh)

def test_supports_match(bank):
    supports = bank.supports
    supports_hz = bank.supports_hz
    for filt_idx in range(bank.num_filts):
        support = supports[filt_idx]
        left_hz, right_hz = supports_hz[filt_idx]
        left_ang = hertz_to_angular(left_hz, bank.sampling_rate)
        right_ang = hertz_to_angular(right_hz, bank.sampling_rate)
        dft_size = int(
            max(1.1 * support, 4 * np.pi / (right_ang - left_ang)))
        freqs = np.arange(dft_size, dtype='float32') / dft_size * 2 * np.pi
        # the "ands" cover [-1, 1] * 2pi periodization.
        zero_mask = np.logical_or(
            np.logical_and(freqs < left_ang, freqs > right_ang - 2 * np.pi),
            np.logical_and(freqs > right_ang, freqs < left_ang + 2 * np.pi),
        )
        if bank.is_real:
            zero_mask[1:] &= zero_mask[-1:0:-1]
        # either there's a zero or the support exceeds 2pi periodization
        # minus one dft bin.
        assert np.any(zero_mask) or \
            right_ang - left_ang >= (2 * np.pi - sum(freqs[:2]))
        assert not np.all(zero_mask), dft_size
        x = bank.get_impulse_response(filt_idx, dft_size)
        # we use 2 * the effective support threshold because of the
        # additive effect when wrapping around the buffer
        if bank.is_zero_phase:
            assert np.allclose(
                x[support // 2 + 1:dft_size - (support // 2) - 1],
                0, atol=2 * EFFECTIVE_SUPPORT_THRESHOLD)
        else:
            assert np.allclose(
                x[support + 1:],
                0, atol=2 * EFFECTIVE_SUPPORT_THRESHOLD)
        X = bank.get_frequency_response(filt_idx, dft_size)
        assert np.allclose(
            X[zero_mask], 0, atol=2 * EFFECTIVE_SUPPORT_THRESHOLD)
