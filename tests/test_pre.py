import numpy as np

from pydrobert.speech.pre import *


def test_dither():
    np.random.seed(1)
    T, std = 100_000, 5
    dither = Dither(std)
    b = dither.apply(np.zeros(T))
    assert b.shape == (T,)
    assert np.isclose(b.std(0), std, atol=1e-2)


def test_preemphasis():
    np.random.seed(2)
    preemph = Preemphasize()
    T = 1028
    a = np.random.rand(T)
    A = np.abs(np.fft.rfft(a))
    A /= A.sum()
    assert A.shape == (T // 2 + 1,)
    hi_A = A[T // 2 + 1 :]
    b = preemph.apply(a)
    assert a.shape == b.shape
    B = np.abs(np.fft.rfft(b))
    B /= B.sum()
    assert A.shape == B.shape
    hi_B = B[T // 2 + 1 :]
    assert np.all(hi_B > hi_A)
