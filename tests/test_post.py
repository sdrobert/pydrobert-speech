# pylint: skip-file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from pydrobert.signal import post

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
def test_standardize_local(norm_var, buff):
    stand = post.Standardize(norm_var=norm_var)
    for axis in range(len(buff.shape)):
        other_axes = tuple(
            idx for idx in range(len(buff.shape)) if idx != axis)
        if sum(buff.shape[idx] for idx in other_axes) == len(other_axes):
            continue
        s_buff = stand.apply(buff, axis=axis)
        assert np.allclose(s_buff.mean(axis=other_axes), 0)
        assert not np.allclose(s_buff, 0)
        if norm_var:
            assert np.allclose(s_buff.var(axis=other_axes), 1), axis

@pytest.mark.parametrize('norm_var', [True, False])
@pytest.mark.parametrize('buff', [
    np.random.random((5, 100)) * np.random.randint(1, 100, 100)
    + np.random.randint(-10, 10, 100),
    np.random.random((8, 1, 10)) * np.random.randint(1, 100, 10)
    + np.random.randint(-10, 10, 10),
    np.random.random((3, 10, 20)) * np.random.randint(1, 100, 20)
    + np.random.randint(-10, 10, 20),
    np.random.random((2, 50, 2, 3)) * np.random.randint(1, 100, 3)
    + np.random.randint(-10, 10, 3),
])
def test_standardize_global(norm_var, buff):
    stand = post.Standardize(norm_var=norm_var)
    for feats in buff:
        stand.accumulate(feats)
    other_axes = tuple(range(len(buff.shape) - 1))
    s_buff_1 = stand.apply(buff)
    assert np.allclose(s_buff_1.mean(axis=other_axes), 0)
    if norm_var:
        assert np.allclose(s_buff_1.var(axis=other_axes), 1)
    # ensure that we're using stored statistics. Otherwise, s_buff_2
    # will have mean zero but not match s_buff_1
    s_buff_2 = stand.apply(buff[0])
    assert np.allclose(s_buff_1[0], s_buff_2)

@pytest.mark.parametrize('do_kaldi', [
    pytest.param(True, marks=pytest.mark.importorskip('pydrobert.kaldi')),
    False,
])
@pytest.mark.parametrize('with_key', [True, False])
def test_standardize_write_read(do_kaldi, with_key, temp_file_1_name):
    if do_kaldi:
        temp_file_1_name = 'ark:' + temp_file_1_name
    stand_1 = post.Standardize()
    x_1 = np.random.random((2, 3, 4))
    x_2 = np.random.random((1, 3, 5)) + np.random.randint(-10, 10)
    x_3 = np.random.random((3, 3, 3)) * 100 - np.random.randint(-10, 10)
    stand_1.accumulate(x_1, axis=1)
    stand_1.accumulate(x_2, axis=1)
    x_1_p_1 = stand_1.apply(x_1, axis=1)
    stand_1.save(temp_file_1_name, key='a')
    stand_1.accumulate(x_3, axis=1)
    x_1_p_2 = stand_1.apply(x_1, axis=1)
    assert not np.allclose(x_1_p_1, x_1_p_2)
    if with_key:
        stand_1.save(temp_file_1_name, key='b')
        stand_2 = post.Standardize(temp_file_1_name, key='b')
    else:
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
