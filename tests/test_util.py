# pylint: skip-file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import erf

import numpy as np
import pytest

from pydrobert.signal import util

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
@pytest.mark.parametrize('do_scipy', [
    pytest.param(True, marks=pytest.mark.importorskip('scipy.norm')),
    False,
])
def test_gauss_quant(mu, std, do_scipy):
    X = np.arange(1000, dtype=float) / 1000 - .5
    X /= X.std()
    X *= std / 2
    X += mu
    for x in X:
        p = .5 * (1 + erf((x - mu) / std / np.sqrt(2)))
        if do_scipy:
            x2 = util.gauss_quant(p, mu=mu, std=std)
        else:
            # because we don't give access to this if scipy is
            # installed, we have to access the private function
            x2 = util._gauss_quant_odeh_evans(p, mu=mu, std=std)
        assert np.isclose(x, x2, atol=1e-5)
