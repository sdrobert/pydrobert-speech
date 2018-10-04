from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from pydrobert.speech import post


@pytest.fixture(params=[
    np.float64,
    np.float32,
    np.int32,
    np.int16
], ids=[
    'f64',
    'f32',
    'i32',
    'i16',
], scope='module')
def dtype(request):
    return request.param


@pytest.mark.parametrize('norm_var', [True, False])
@pytest.mark.parametrize('buff', [
    x * np.random.randint(1, 100) + np.random.randint(-10, 10)
    for x in [
        np.random.random((100, 1)),
        np.random.random((1, 10)),
        np.random.random((5, 5)),
        np.random.random((10, 4, 3)),
    ]
])
def test_standardize_local(norm_var, buff, dtype):
    if np.allclose(buff, buff[:1].ravel()[0]):
        pytest.skip()
    buff = buff.astype(dtype)
    stand = post.Standardize(norm_var=norm_var)
    for axis in range(len(buff.shape)):
        buff_2 = buff.copy()
        other_axes = tuple(
            idx for idx in range(len(buff.shape)) if idx != axis)
        if sum(buff_2.shape[idx] for idx in other_axes) == len(other_axes):
            continue
        # we make sure that one sample along the target axis is
        # different from at least one other
        s_1 = [0] * len(buff_2.shape)
        s_2 = [-1] * len(buff_2.shape)
        s_1[axis] = slice(None)
        s_2[axis] = slice(None)
        s_1 = tuple(s_1)
        s_2 = tuple(s_2)
        buff_2[s_1] = buff_2[s_2] - 1
        assert not np.any(np.isclose(buff_2.std(axis=other_axes), 0))
        s_buff = stand.apply(buff_2, axis=axis)
        assert np.allclose(s_buff.mean(axis=other_axes), 0)
        assert not np.allclose(s_buff, 0)
        if norm_var:
            assert np.allclose(s_buff.var(axis=other_axes), 1), axis


@pytest.mark.parametrize('norm_var', [True, False])
@pytest.mark.parametrize('buff', [
    np.random.random((5, 100)) * np.random.randint(1, 100, 100) + (
        np.random.randint(-10, 10, 100)),
    np.random.random((8, 1, 10)) * np.random.randint(1, 100, 10) + (
        + np.random.randint(-10, 10, 10)),
    np.random.random((3, 10, 20)) * np.random.randint(1, 100, 20) + (
        + np.random.randint(-10, 10, 20)),
    np.random.random((2, 50, 2, 3)) * np.random.randint(1, 100, 3) + (
        + np.random.randint(-10, 10, 3)),
])
def test_standardize_global(norm_var, buff, dtype):
    buff = buff.astype(dtype)
    if np.allclose(buff, buff[:1].ravel()[0]):
        pytest.skip()
    other_axes = tuple(range(len(buff.shape) - 1))
    if norm_var:
        # quick fix when ints misbehave and give zero variance
        if np.any(np.isclose(buff.std(axis=other_axes), 0)):
            buff = np.zeros(buff.shape, dtype=buff.dtype)
            buff[0, ...] = 1
    stand = post.Standardize(norm_var=norm_var)
    for feats in buff:
        stand.accumulate(feats)
    s_buff_1 = stand.apply(buff)
    assert np.allclose(s_buff_1.mean(axis=other_axes), 0)
    if norm_var:
        assert np.allclose(s_buff_1.var(axis=other_axes), 1)
    # ensure that we're using stored statistics. Otherwise, s_buff_2
    # will have mean zero but not match s_buff_1
    s_buff_2 = stand.apply(buff[0])
    assert np.allclose(s_buff_1[0], s_buff_2)


def test_standardize_write_read(temp_file_1_name):
    stand_1 = post.Standardize()
    x_1 = np.random.random((2, 3, 4))
    x_2 = np.random.random((1, 3, 5)) + np.random.randint(-10, 10)
    x_3 = np.random.random((3, 3, 3)) * 100 - np.random.randint(-10, 10)
    stand_1.accumulate(x_1, axis=1)
    stand_1.accumulate(x_2, axis=1)
    x_1_p_1 = stand_1.apply(x_1, axis=1)
    stand_1.save(temp_file_1_name)
    stand_1.accumulate(x_3, axis=1)
    x_1_p_2 = stand_1.apply(x_1, axis=1)
    assert not np.allclose(x_1_p_1, x_1_p_2)
    stand_1.save(temp_file_1_name)
    stand_2 = post.Standardize(temp_file_1_name)
    x_1_p_3 = stand_2.apply(x_1, axis=1)
    assert np.allclose(x_1_p_2, x_1_p_3)


@pytest.mark.parametrize('buff', [
    np.random.random(10),
    np.random.random((2, 5)),
    np.random.random((3, 6, 4)),
    np.random.random((5, 4, 0, 0, 1)),
])
@pytest.mark.parametrize('concatenate', [True, False])
@pytest.mark.parametrize('num_deltas', list(range(5)))
def test_delta_shapes(buff, concatenate, num_deltas):
    for target_axis in range(len(buff.shape) + 1 - int(concatenate)):
        deltas = post.Deltas(
            num_deltas, concatenate=concatenate, target_axis=target_axis)
        for axis in range(len(buff.shape)):
            new_shape = list(buff.shape)
            if concatenate:
                new_shape[target_axis] *= num_deltas + 1
            else:
                new_shape.insert(target_axis, num_deltas + 1)
            assert deltas.apply(buff, axis=axis).shape == tuple(new_shape), \
                buff.shape


class KaldiDeltas(object):
    '''Replicate Kaldi delta logic for comparative purposes'''

    def __init__(self, num_deltas, window=2):
        self._scales = [np.ones(1, dtype=np.float64)]
        for last_idx in range(num_deltas):
            prev_scale = self._scales[last_idx]
            cur_scale = np.zeros(
                len(prev_scale) + window * 2, dtype=np.float64)
            prev_offset = (len(prev_scale) - 1) // 2
            cur_offset = prev_offset + window
            normalizer = 0
            for j in range(-window, window + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    cur_scale[j + k + cur_offset] += \
                        j * prev_scale[k + prev_offset]
            cur_scale /= normalizer
            self._scales.append(cur_scale)

    def _process(self, r, features, out_row):
        features = features.astype(np.float64, copy=False)
        num_frames, feat_dim = features.shape
        assert len(out_row) == feat_dim * len(self._scales)
        for idx, scale in enumerate(self._scales):
            max_offset = (len(scale) - 1) // 2
            sub_row = out_row[idx * feat_dim:(idx + 1) * feat_dim]
            for j in range(-max_offset, max_offset + 1):
                offset_frame = r + j
                if offset_frame < 0:
                    offset_frame = 0
                elif offset_frame >= num_frames:
                    offset_frame = num_frames - 1
                sub_row += scale[j + max_offset] * features[offset_frame]

    def apply(self, features):
        assert len(features.shape) == 2
        out = np.zeros(
            (features.shape[0], features.shape[1] * len(self._scales)),
            dtype=np.float64
        )
        for r, out_row in enumerate(out):
            self._process(r, features, out_row)
        return out.astype(features.dtype, copy=False)


@pytest.mark.parametrize('buff', [
    np.random.random((1, 3)),
    np.random.random((3, 1)),
    np.random.random((20, 50)),
])
@pytest.mark.parametrize('num_deltas', list(range(5)))
@pytest.mark.parametrize('window', list(range(1, 6)))
def test_compare_to_kaldi(buff, num_deltas, window, dtype):
    buff = buff.astype(dtype)
    deltas = post.Deltas(
        num_deltas, concatenate=True, context_window=window, target_axis=1)
    kaldi_deltas = KaldiDeltas(num_deltas, window)
    delta_res = deltas.apply(buff, axis=0)
    kaldi_res = kaldi_deltas.apply(buff)
    assert np.allclose(delta_res, kaldi_res)
