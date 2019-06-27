# pylint: skip-file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import warnings

from math import erf
from tempfile import mkdtemp

import numpy as np
import pytest

from pydrobert.speech import util


@pytest.mark.parametrize('shift', [0, 1, 100, -100])
@pytest.mark.parametrize(
    'dft_size', [1, 2, 51, 1000], ids=['l1', 'l2', 'l51', 'l1000'])
@pytest.mark.parametrize('copy', [True, False], ids=['copy', 'keep'])
@pytest.mark.parametrize(
    'start_idx', [0, 1, -1], ids=['at', 'after', 'before'])
def test_circshift_fourier(shift, dft_size, start_idx, copy):
    start_idx %= dft_size
    zeros = np.random.randint(dft_size)
    X = 10 * np.random.random(dft_size - zeros) + \
        10j * np.random.random(dft_size - zeros)
    Xs = util.circshift_fourier(
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


@pytest.mark.parametrize('mu', [0, -1, 100])
@pytest.mark.parametrize('std', [.1, 1, 10])
@pytest.mark.parametrize('do_scipy', [True, False])
def test_gauss_quant(mu, std, do_scipy):
    X = np.arange(1000, dtype=float) / 1000 - .5
    X /= X.std()
    X *= std / 2
    X += mu
    for x in X:
        p = .5 * (1 + erf((x - mu) / std / np.sqrt(2)))
        if do_scipy:
            pytest.importorskip('scipy.norm')
            x2 = util.gauss_quant(p, mu=mu, std=std)
        else:
            # because we don't give access to this if scipy is
            # installed, we have to access the private function
            x2 = util._gauss_quant_odeh_evans(p, mu=mu, std=std)
        assert np.isclose(x, x2, atol=1e-5)


@pytest.mark.parametrize('key', [True, False])
def test_read_kaldi(temp_dir, key):
    kaldi = pytest.importorskip('pydrobert.kaldi.io')
    rxfilename = 'ark:{}'.format(os.path.join(temp_dir, 'foo.ark'))
    key_1 = 'lions'
    key_2 = 'tigers'
    buff_1 = np.random.random((100, 10))
    buff_2 = np.random.random((1000, 2))
    with kaldi.open(rxfilename, 'dm', 'w') as table:
        table.write(key_1, buff_1)
        table.write(key_2, buff_2)
    if key:
        buff_3 = util.read_signal(rxfilename, dtype='dm', key=key_2)
        assert np.allclose(buff_2, buff_3)
    else:
        buff_3 = util.read_signal(rxfilename, dtype='dm')
        assert np.allclose(buff_1, buff_3)


@pytest.mark.parametrize('use_scipy', [True, False])
@pytest.mark.parametrize('channels', [1, 2], ids=['mono', 'stereo'])
@pytest.mark.parametrize('sampwidth', [2, 4])
def test_read_wave(temp_dir, use_scipy, channels, sampwidth):
    import wave
    rfilename = os.path.join(temp_dir, 'foo.wav')
    if channels > 1:
        wave_buffer_1 = np.random.random((1000, channels)) * 1000
    else:
        wave_buffer_1 = np.random.random(1000) * 1000
    wave_buffer_1 = wave_buffer_1.astype('<i{}'.format(sampwidth))
    wave_bytes = wave_buffer_1.tobytes('C')
    wave_file = wave.open(rfilename, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(sampwidth)
    wave_file.setframerate(8000)
    wave_file.writeframes(wave_bytes)
    wave_file.close()
    if use_scipy:
        pytest.importorskip('scipy')
        wave_buffer_2 = util._scipy_io_read_signal(rfilename, None, None)
    else:
        wave_buffer_2 = util._wave_read_signal(rfilename, None, None)
    assert np.allclose(wave_buffer_1, wave_buffer_2)


@pytest.mark.parametrize('key', [True, False])
def test_read_hdf5(temp_dir, key):
    h5py = pytest.importorskip('h5py')
    rfilename = os.path.join(temp_dir, 'foo.hdf5')
    h5py_file = h5py.File(rfilename, 'w')
    h5py_file.create_group('a/b/c')
    h5py_file.create_group('a/b/d/e')
    dset_1 = np.random.random((1000, 2000))
    dset_2 = (np.random.random(10) * 1000).astype(int)
    h5py_file.create_dataset('a/b/d/f', (1000, 2000), data=dset_1)
    h5py_file.create_dataset('g', (10,), data=dset_2)
    h5py_file.close()
    if key:
        dset_3 = util.read_signal(rfilename, key='g')
        assert np.allclose(dset_2, dset_3)
    else:
        dset_3 = util.read_signal(rfilename)
        assert np.allclose(dset_1, dset_3)


def test_read_torch(temp_dir):
    torch = pytest.importorskip('torch')
    torch.manual_seed(10)
    rfilename = 'foo.pt'
    exp = torch.randn(10, 4)
    torch.save(exp, rfilename)
    exp = exp.numpy()
    act = util.read_signal(rfilename)
    assert np.allclose(exp, act)


@pytest.mark.parametrize(
    'allow_pickle', [True, False], ids=['picklable', 'notpicklable'])
@pytest.mark.parametrize('fix_imports', [True, False], ids=['fix', 'nofix'])
def test_read_numpy_binary(temp_dir, allow_pickle, fix_imports):
    rfilename = os.path.join(temp_dir, 'foo.npy')
    buff_1 = np.random.random((1000, 10, 5))
    np.save(
        rfilename, buff_1, allow_pickle=allow_pickle, fix_imports=fix_imports)
    buff_2 = util.read_signal(rfilename)
    assert np.allclose(buff_1, buff_2)


@pytest.mark.parametrize(
    'compressed', [True, False], ids=['compressed', 'uncompressed'])
@pytest.mark.parametrize('key', [True, False], ids=['withkey', 'withoutkey'])
def test_read_numpy_archive(temp_dir, compressed, key):
    rfilename = os.path.join(temp_dir, 'foo.npz')
    buff_1 = np.random.random((5, 1, 2))
    buff_2 = np.random.random((1,))
    if compressed and key:
        np.savez_compressed(rfilename, a=buff_1, b=buff_2)
    elif compressed:
        np.savez_compressed(rfilename, buff_1, buff_2)
    elif key:
        np.savez(rfilename, a=buff_1, b=buff_2)
    else:
        np.savez(rfilename, buff_1, buff_2)
    if key:
        buff_3 = util.read_signal(rfilename, key='a')
    else:
        buff_3 = util.read_signal(rfilename)
    assert np.allclose(buff_1, buff_3)


@pytest.mark.parametrize('text', [True, False])
def test_read_numpy_fromfile(temp_dir, text):
    rfilename = os.path.join(temp_dir, 'foo')
    buff_1 = np.random.random(1000)
    sep = "," if text else ""
    buff_1.tofile(rfilename, sep=sep)
    buff_2 = util.read_signal(rfilename, sep=sep)
    assert np.allclose(buff_1, buff_2)
