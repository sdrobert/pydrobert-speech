from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from json import load as json_load
from wave import open as wave_open

import numpy as np
import pydrobert.speech.compute as compute
import pydrobert.speech.config as config
import pytest

from pydrobert.speech.util import alias_factory_subclass_from_arg
from six.moves.cPickle import load as pickle_load


@pytest.fixture(params=[
    'causal',
    'centered',
], scope='module',)
def frame_style(request):
    return request.param


@pytest.fixture(params=[
    lambda frame_style: compute.STFTFrameComputer(
        {'name': 'gabor', 'scaling_function': 'mel'},
        frame_length_ms=25,
        frame_shift_ms=10,
        use_power=True,
        use_log=True,
        pad_to_nearest_power_of_two=np.random.randint(2),
        include_energy=np.random.randint(2),
        frame_style=frame_style
    ),
    lambda frame_style: compute.SIFrameComputer(
        {'name': 'gabor', 'scaling_function': 'mel'},
        frame_shift_ms=25,
        use_power=True,
        use_log=True,
        pad_to_nearest_power_of_two=np.random.randint(2),
        include_energy=np.random.randint(2),
        frame_style=frame_style,
    ),
], ids=[
    'stft',
    'si',
],)
def computer(request, frame_style):
    return request.param(frame_style)


@pytest.fixture(params=[
    0,
    1,
    2 ** 8,
    2 ** 10,
], ids=[
    'empty buffer',
    'length 1 buffer',
    'medium buffer',
    'large buffer',
], scope="module")
def buff(request):
    b = np.random.random(request.param)
    b.flags.writeable = False
    return b


def test_framewise_matches_full(computer, buff):
    feats_full = computer.compute_full(buff)
    feats_framewise = compute.frame_by_frame_calculation(computer, buff)
    assert np.allclose(feats_full, feats_framewise), (
        feats_full.shape[0],
        np.where(
            np.logical_not(np.isclose(feats_full, feats_framewise)))[0],
    )


def test_chunk_sizes_dont_matter_to_result(computer, buff):
    feats = compute.frame_by_frame_calculation(computer, buff)
    feats_chunks = []
    while len(buff):
        next_len = np.random.randint(len(buff) + 1)
        feats_chunks.append(computer.compute_chunk(buff[:next_len]))
        buff = buff[next_len:]
    feats_chunks.append(computer.finalize())
    assert np.allclose(feats, np.concatenate(feats_chunks)), (
        feats.shape[0],
        np.where(np.logical_not(np.isclose(
            feats, np.concatenate(feats_chunks)
        )))
    )


def test_zero_samples_generate_zero_features(computer):
    assert computer.compute_full(np.empty(0)).shape == (0, computer.num_coeffs)
    assert computer.compute_chunk(np.empty(0)).shape == (
        0, computer.num_coeffs)
    assert computer.finalize().shape == (0, computer.num_coeffs)


def test_finalize_twice_generates_no_coefficients(computer):
    buff = np.random.random(computer.frame_length * 2)
    buff.flags.writeable = False
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
    assert np.allclose(
        compute.frame_by_frame_calculation(computer, buff),
        compute.frame_by_frame_calculation(computer, buff)
    )


class TestFFTPACK(object):

    def setup_method(self):
        pytest.importorskip('scipy')
        self._orig_USE_FFTPACK = config.USE_FFTPACK

    def test_computations_same_between_numpy_scipy(self, computer, buff):
        config.USE_FFTPACK = False
        np_feats = computer.compute_full(buff)
        config.USE_FFTPACK = True
        scipy_feats = computer.compute_full(buff)
        assert np.allclose(np_feats, scipy_feats)

    def teardown_method(self):
        config.USE_FFTPACK = self._orig_USE_FFTPACK


class SIFrameComputerNpConvolve(compute.SIFrameComputer):
    '''For compute_full, use np.convolve instead of overlap-save'''

    aliases = {}

    def __init__(self, *args, **kwargs):
        super(SIFrameComputerNpConvolve, self).__init__(*args, **kwargs)
        # convert filters back to impulse responses...
        for filt_idx in range(len(self._filts)):
            ifilt = self._compute_idft(self._filts[filt_idx])
            assert np.allclose(ifilt[self._max_support + 1:], 0)
            self._filts[filt_idx] = ifilt[:self._max_support]
        self._window = self._window.flatten()

    def compute_full(self, signal):
        frame_length = self._frame_length
        frame_shift = self._frame_shift
        num_frames = (len(signal) + frame_shift // 2)
        num_frames //= frame_shift
        coeffs = np.empty((num_frames, self.num_coeffs), dtype=signal.dtype)
        if not num_frames:
            return coeffs
        signal = np.pad(
            signal,
            (
                max(0, frame_shift - self._translation),
                frame_length + self._translation,  # liberal
            ),
            'constant',
        )
        for coeff_idx, filt in enumerate(self._filts):
            y = np.convolve(signal, filt)
            if self._power:
                y[:] = y * y.conj()
            else:
                y[:] = np.abs(y)
            if self._frame_style == 'centered':
                frame_start = max(0, self._translation - frame_shift)
            else:
                frame_start = self._translation
            for frame_idx in range(num_frames):
                frame_end = frame_start + 2 * self._frame_shift
                frame = y[frame_start:frame_end]
                coeffs[frame_idx, coeff_idx] = np.sum(
                    frame.real * self._window)
                frame_start += self._frame_shift
        if self._log:
            coeffs[:] = np.log(np.maximum(coeffs, config.LOG_FLOOR_VALUE))
        return coeffs


def test_overlap_save_convolve_the_same(buff):
    os_computer = compute.SIFrameComputer(
        {'name': 'gabor', 'scaling_function': 'mel'})
    conv_computer = SIFrameComputerNpConvolve(
        {'name': 'gabor', 'scaling_function': 'mel'})
    os_coeffs = os_computer.compute_full(buff)
    conv_coeffs = conv_computer.compute_full(buff)
    assert os_coeffs.shape == conv_coeffs.shape
    assert np.allclose(os_coeffs, conv_coeffs)


def test_kaldi_comp_matches_fbank_comp():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    with open(os.path.join(data_dir, 'kaldi_feats.pkl'), 'rb') as pkl_file:
        kaldi_feats = pickle_load(pkl_file)
    with open(os.path.join(data_dir, 'noise.pkl'), 'rb') as pkl_file:
        sig = pickle_load(pkl_file)
    with open(os.path.join(data_dir, 'fbank.json')) as json_file:
        computer = alias_factory_subclass_from_arg(
            compute.FrameComputer, json_load(json_file))
    fbank_feats = computer.compute_full(sig)
    # remove unit-normalization of hann window. We use the power spectrum, so
    # the denominator was squared, hence the 2x in the log domain
    fbank_feats += 2 * np.log(0.5 * (computer.frame_length - 1))
    # removes x2 for real-spectrum output. We double because of the
    # hermitian symmetry of real-valued signals; kaldi doesn't
    fbank_feats -= np.log(2)
    assert fbank_feats.shape == kaldi_feats.shape
    assert np.allclose(fbank_feats, kaldi_feats)
