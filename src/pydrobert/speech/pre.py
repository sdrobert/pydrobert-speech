# Copyright 2021 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classes for pre-processing speech signals"""


import abc
from typing import Optional

import numpy as np

from pydrobert.speech.alias import AliasedFactory

__all__ = [
    "PreProcessor",
    "Dither",
    "Preemphasize",
]


class PreProcessor(AliasedFactory):
    """A container for pre-processing signals with a transform"""

    @abc.abstractmethod
    def apply(self, signal: np.ndarray, axis: int = -1, in_place=False) -> np.ndarray:
        """Applies the transformation to a signal tensor

        Consult the class documentation for more details on what the transformation is.

        Parameters
        ----------
        signal
        axis
            The axis of `signal` to apply the transformation along
        in_place
            Whether it is okay to modify `signal` (:obj:`True`) or whether a copy should
            be made (:obj:`False`)

        Returns
        -------
        out : np.ndarray
            The transformed features
        """
        pass


class Dither(PreProcessor):
    """Add random noise to a signal tensor

    The default axis of `apply` has been set to None, which will generate random noise
    for each coefficient. This is likely the desired behaviour. Setting axis to an
    integer will add random values along 1D slices of that axis.

    Intermediate values are calculated as 64-bit floats. The result is cast back to the
    input data type.

    Parameters
    ----------
    coeff
        Added noise will be in the range ``[-coeff, coeff]``
    """

    coeff: float  #:
    aliases = {"dither", "dithering"}  #:

    def __init__(self, coeff: float = 1.0):
        super().__init__()
        self.coeff = coeff

    def apply(
        self, signal: np.ndarray, axis: Optional[int] = None, in_place: bool = False
    ) -> np.ndarray:
        signal_dtype = signal.dtype
        if not in_place or signal.dtype != np.float64:
            signal = signal.astype(np.float64)
        if axis is None or not signal.shape or len(signal.shape) == 1:
            signal += self.coeff * np.random.random(signal.shape)
        else:
            random_shape = [1] * len(signal.shape)
            random_shape[axis] = signal.shape[axis]
            signal += self.coeff * np.random.random(random_shape)
        return signal.astype(signal_dtype, copy=False)


class Preemphasize(PreProcessor):
    """Attenuate the low frequencies of a signal by taking sample differences

    The following transformation is applied along the target axis

    ::

        new[i] = old[i] - coeff * old[i-1] for i > 1
        new[0] = old[0]

    This is essentially a convolution with a Haar wavelet for positive `coeff`. It
    emphasizes high frequencies.

    Intermediate values are calculated as 64-bit floats. The result is cast back to the
    input data type.

    Parameters
    ----------
    coeff
    """

    coeff: float  #:
    aliases = {"preemphasize", "preemphasis", "preemph"}  #:

    def __init__(self, coeff: float = 0.97):
        super().__init__()
        self.coeff = coeff

    def apply(
        self, signal: np.ndarray, axis: int = -1, in_place: bool = False
    ) -> np.ndarray:
        signal_dtype = signal.dtype
        if not in_place or signal_dtype != np.float64:
            signal = signal.astype(np.float64)
        if signal.shape and len(signal.shape) > 1:
            signal[1:] -= self.coeff * signal[:-1]
        else:
            tensor_slice_mind = [slice(None)] * len(signal.shape)
            tensor_slice_subhd = [slice(None)] * len(signal.shape)
            tensor_slice_mind[axis] = slice(1, None)
            tensor_slice_subhd[axis] = slice(-1, None)
            signal[tensor_slice_mind] -= self.coeff * signal[tensor_slice_subhd]
        return signal.astype(signal_dtype, copy=False)
