# pylint: skip-file

import numpy as np
import pytest

from pydrobert.signal import banks

def test_scales_invertible(scaling_function):
    for hertz in range(20, 8000):
        scale = scaling_function.hertz_to_scale(hertz)
        assert np.isclose(hertz, scaling_function.scale_to_hertz(scale)), \
                "Inverse not equal to orig for {} at {}".format(
                    scaling_function,
                    hertz,
                )

@pytest.mark.parametrize('shift', [0, 1, 100, -100])
@pytest.mark.parametrize(
    'dft_size', [1, 2, 51, 1000], ids=['l1', 'l2', 'l51', 'l1000'])
@pytest.mark.parametrize('copy', [True, False], ids=['copy', 'keep'])
@pytest.mark.parametrize(
    'start_idx', [0, 1, -1], ids=['at', 'after', 'before'])
def test_circshift_fourier(shift, dft_size, start_idx, copy):
    start_idx %= dft_size
    zeros = np.random.randint(dft_size)
    X = 10 * np.random.random(dft_size - zeros) + 10j * np.random.random(dft_size - zeros)
    Xs = banks.circshift_fourier(
        X.copy(),
        shift,
        start_idx=start_idx,
        dft_size=dft_size,
        copy=copy
    )
    X = np.roll(np.pad(X, (0, zeros), 'constant'), start_idx)
    Xs = np.roll(np.pad(Xs, (0, zeros), mode='constant'), start_idx)
    assert len(X) == len(Xs)
    x = np.fft.ifft(X)
    xs = np.fft.ifft(Xs)
    assert np.allclose(np.roll(x, shift), xs)

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
        assert np.allclose(
            full_response, challenge,
            atol=banks.EFFECTIVE_SUPPORT_THRESHOLD), \
                '{} {}'.format(bin_idx, len(truncated))

def test_frequency_matches_impulse(bank):
    for filt_idx in range(bank.num_filts):
        dft_size = int(max(
            # allow over- or under-sampling
            bank.supports[filt_idx] * (1 + np.random.random()),
            2 * bank.sampling_rate / bank.supports_hz[filt_idx],
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
    centers_hz = bank.centers_hz
    for filt_idx in range(bank.num_filts):
        support = supports[filt_idx]
        support_hz = supports_hz[filt_idx]
        center_hz = centers_hz[filt_idx]
        left_ang = banks.hertz_to_angular(
            center_hz - support_hz / 2,
            bank.sampling_rate,
        )
        right_ang = banks.hertz_to_angular(
            center_hz + support_hz / 2,
            bank.sampling_rate,
        )
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
        assert np.any(zero_mask) or \
            right_ang - left_ang >= (2 * np.pi - freqs[1])
        assert not np.all(zero_mask), dft_size
        x = bank.get_impulse_response(filt_idx, dft_size)
        # we use 2 * the effective support threshold because of the
        # additive effect when wrapping around the buffer
        if bank.is_zero_phase:
            assert np.allclose(
                x[support // 2 + 1:dft_size - (support // 2) - 1],
                0, atol=2 * banks.EFFECTIVE_SUPPORT_THRESHOLD)
        else:
            assert np.allclose(
                x[support + 1:],
                0, atol=2 * banks.EFFECTIVE_SUPPORT_THRESHOLD)
        X = bank.get_frequency_response(filt_idx, dft_size)
        assert np.allclose(
            X[zero_mask], 0, atol=2 * banks.EFFECTIVE_SUPPORT_THRESHOLD)

def test_framewise_matches_full(computer, buff):
    feats_full = computer.compute_full(buff)
    feats_framewise = banks.frame_by_frame_calculation(computer, buff)
    assert np.allclose(feats_full, feats_framewise), \
            np.where(
                np.logical_not(np.isclose(feats_full, feats_framewise)))[0]

def test_chunk_sizes_dont_matter_to_result(computer, buff):
    feats = banks.frame_by_frame_calculation(computer, buff)
    feats_chunks = []
    while len(buff):
        next_len = np.random.randint(len(buff) + 1)
        feats_chunks.append(computer.compute_chunk(buff[:next_len]))
        buff = buff[next_len:]
    feats_chunks.append(computer.finalize())
    assert np.allclose(feats, np.concatenate(feats_chunks)), \
        (
            feats.shape[0],
            np.where(np.logical_not(np.isclose(
                feats, np.concatenate(feats_chunks)
            )))
        )

def test_zero_samples_generate_zero_features(computer):
    assert computer.compute_full(np.empty(0)).shape \
            == (0, computer.num_coeffs)
    assert computer.compute_chunk(np.empty(0)).shape \
            == (0, computer.num_coeffs)
    assert computer.finalize().shape == (0, computer.num_coeffs)

def test_finalize_twice_generates_no_coefficients(computer):
    buff = np.empty(computer.frame_length * 2, np.float64)
    coeffs = np.concatenate([
        computer.compute_chunk(buff),
        computer.finalize()
    ])
    assert coeffs.shape[0] >= 1
    assert computer.finalize().shape == (0, computer.num_coeffs)

def test_started_makes_sense(computer):
    assert not computer.started
    computer.compute_chunk(np.empty(1))
    assert computer.started
    computer.finalize()
    assert not computer.started

def test_repeated_calls_generate_same_results(computer, buff):
    assert np.allclose(
        computer.compute_full(buff), computer.compute_full(buff))

@pytest.mark.skipif(not banks.USE_FFTPACK, reason='fftpack disabled')
class TestFFTPACK(object):

    def setup_method(self):
        self._orig_USE_FFTPACK = banks.USE_FFTPACK

    def test_computations_same_between_numpy_scipy(self, computer, buff):
        banks.USE_FFTPACK = False
        np_feats = computer.compute_full(buff)
        banks.USE_FFTPACK = True
        scipy_feats = computer.compute_full(buff)
        assert np.allclose(np_feats, scipy_feats)

    def teardown_method(self):
        banks.USE_FFTPACK = self._orig_USE_FFTPACK
