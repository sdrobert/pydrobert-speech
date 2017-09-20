# pylint: skip-file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from pydrobert.signal import config
from pydrobert.signal.compute import frame_by_frame_calculation

@pytest.fixture(params=[
    0,
    1,
    2 ** 8,
], ids=[
    'empty buffer',
    'length 1 buffer',
    'large buffer',
], scope="module")
def buff(request):
    b = np.random.random(request.param)
    b.flags.writeable = False
    return b

def test_framewise_matches_full(computer, buff):
    feats_full = computer.compute_full(buff)
    feats_framewise = frame_by_frame_calculation(computer, buff)
    assert np.allclose(feats_full, feats_framewise), \
        (
            feats_full.shape[0],
            np.where(
                np.logical_not(np.isclose(feats_full, feats_framewise)))[0],
        )

def test_chunk_sizes_dont_matter_to_result(computer, buff):
    feats = frame_by_frame_calculation(computer, buff)
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
        frame_by_frame_calculation(computer, buff),
        frame_by_frame_calculation(computer, buff)
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
