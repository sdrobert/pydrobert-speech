# Copyright 2023 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch compatibility module

This submodule is intended to provide PyTorch implementations of the components critical
to feature computation. It is not meant to comprehensively reproduce all functionality
in PyTorch. Each PyTorch module here contains a class method which initializes the
PyTorch module with some analogous Numpy instance discussed elsewhere. For example,
assuming `stft_frame_computer` is an instance of a
:class:`pydrobert.speech.STFTFrameComputer`, one may instantiate a
:class:`PyTorchSTFTFrameComputer` via

>>> pytorch_stft_frame_computer = PyTorchSTFTFrameComputer.from_stft_frame_computer(
...     stft_frame_computer)
"""

import math

from typing_extensions import Self, Literal
from typing import Collection, List, Optional, Sequence, Tuple

import torch

from . import config
from .pre import Dither, Preemphasize
from .compute import STFTFrameComputer

__all__ = [
    "pytorch_dither",
    "pytorch_preemphasize",
    "pytorch_stft_frame_computer",
    "PyTorchDither",
    "PyTorchPreemphasize",
    "PyTorchSTFTFrameComputer",
]


def check_in(name: str, val: str, choices: Collection[str]):
    if val not in choices:
        choices = "', '".join(sorted(choices))
        raise ValueError(f"Expected {name} to be one of '{choices}'; got '{val}'")


def check_positive(name: str, val, nonnegative=False):
    pos = "non-negative" if nonnegative else "positive"
    if val < 0 or (val == 0 and not nonnegative):
        raise ValueError(f"Expected {name} to be {pos}; got {val}")


def pytorch_preemphasize(sig: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """Functional implementation of PyTorchPreemphasize"""
    sig = torch.concatenate([sig.new_zeros(1), sig])
    return sig[1:] - coeff * sig[:-1]


class PyTorchPreemphasize(torch.nn.Module):
    """PyTorch implementation of Preemphasize
    
    Parameters
    ----------
    coeff
        Preemphasis coefficient
    """

    __constants__ = ("coeff",)
    coeff: float

    def __init__(self, coeff: float = 0.97) -> None:
        super().__init__()
        self.coeff = coeff

    @classmethod
    def from_preemphasize(cls, preemphasize: Preemphasize) -> Self:
        return cls(preemphasize.coeff)

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        return pytorch_preemphasize(sig, self.coeff)


def pytorch_dither(sig: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
    """Functional implementation of PyTorchDither"""
    return sig + coeff * torch.randn_like(sig)


class PyTorchDither(torch.nn.Module):
    """PyTorch implementation of Dither
    
    Add random, normally-distributed noise to a signal

    Parameters
    ----------
    coeff
        The standard deviation of the noise
    dim
        The dimension to apply noise to. If unspecified, applied to all coefficients
    
    Notes
    -----
    While it is usually the case in PyTorch that random noise is only added during
    training, dithering serves a
    """

    __constants__ = ("coeff",)
    coeff: float

    def __init__(self, coeff: float = 1.0):
        check_positive("coeff", coeff, True)
        super().__init__()
        self.coeff = coeff

    @classmethod
    def from_dither(cls, dither: Dither) -> Self:
        return cls(dither.coeff)

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        return pytorch_dither(sig, self.coeff)


@torch.jit.script_if_tracing
def pytorch_stft_frame_computer(
    sig: torch.Tensor,
    filters: List[torch.Tensor],
    offsets: List[int],
    frame_length: int,
    frame_shift: int,
    centered: bool = True,
    window: Optional[torch.Tensor] = None,
    dft_size: Optional[int] = None,
    use_log: bool = True,
    use_power: bool = False,
    include_energy: bool = False,
    kaldi_shift: bool = False,
    is_real: bool = True,
    eps: float = config.LOG_FLOOR_VALUE,
) -> torch.Tensor:
    """Functional implementation of PyTorchSTFTFrameComputer"""
    if dft_size is None:
        dft_size_ = int(2 ** math.ceil(math.log(frame_length, 2)))
    elif dft_size < frame_length:
        raise RuntimeError(f"expected dft_size gte {frame_length}; got {dft_size}")
    else:
        dft_size_ = dft_size
    num_filts = len(filters)
    if num_filts != len(offsets):
        raise RuntimeError(
            f"filters ({num_filts}) has different length than offsets "
            f"({len(offsets)})"
        )
    if sig.ndim != 1:
        raise RuntimeError(f"Expected x to be 1-dimensional; got {sig.ndim}")
    if window is not None and window.shape != (frame_length,):
        raise RuntimeError(
            f"Expected window to have shape {(frame_length,)}; got {window.shape}"
        )
    sig_len = sig.size(0)
    if sig_len < frame_length // 2 + 1:
        return sig.new_empty((0, num_filts))
    zero = sig.new_zeros(1)
    if not centered:
        pad_left = 0
    elif kaldi_shift:
        pad_left = frame_length // 2 - frame_shift // 2
    else:
        pad_left = (frame_length + 1) // 2 - 1
    num_frames = max(0, (sig_len + frame_shift // 2) // frame_shift)
    total_len = (num_frames - 1) * frame_shift - pad_left + frame_length
    pad_right = max(0, total_len - sig_len)
    if pad_left or pad_right:
        # symmetric padding
        sig = torch.cat(
            [sig[:pad_left].flip(0), sig, sig[sig_len - pad_right :].flip(0)]
        )
    sig = sig.as_strided((num_frames, frame_length), (frame_shift, 1))
    y: List[torch.Tensor] = []
    if include_energy:
        energy = torch.linalg.norm(sig, 2, 1) / math.sqrt(frame_length)
        if use_power:
            energy = energy.square()
        y.append(energy)
    if window is not None:
        sig = sig * window
    spect = torch.fft.rfft(sig, dft_size_, 1, "backward")
    del sig
    half_len = spect.size(1)
    mod = half_len % 2
    for si, filt in zip(offsets, filters):
        val, consumed, conj, filt_len = zero, 0, False, len(filt)
        while consumed < filt_len:
            if conj:
                seg_len = max(min(si + filt_len - consumed, half_len - 2 + mod) - si, 0)
                seg = spect[..., -2 + mod - si - seg_len : -2 + mod - si].conj().flip(1)
                si -= half_len - 2 + mod
            else:
                seg_len = max(0, min(si + filt_len - consumed, half_len) - si)
                seg = spect[..., si : si + seg_len]
                si -= half_len
            seg = seg * filt[consumed : consumed + seg_len]
            if use_power:
                val_f = torch.linalg.norm(seg, 2, 1).square()
            else:
                val_f = seg.abs().sum(1)
            if is_real:
                val_f = val_f * 2
            val = val + val_f
            conj = not conj
            consumed += seg_len
            si = max(0, si)
        y.append(val)
    y_ = torch.stack(y, 1)
    if use_log:
        y_ = y_.clamp_min(eps).log()
    return y_


class PyTorchSTFTFrameComputer(torch.nn.Module):
    """PyTorch implementation of STFTFrameComputer

    This module is a port of :class:`pydrobert.speech.compute.STFTFrameComputer` to
    PyTorch. When called, the output should be nearly identical to a call to
    :func:`STFTFrameComputer.compute_full`, except :class:`torch.Tensor` inputs and
    outputs are expected.

    The easiest means of initializing this module is through the factory function
    :func:`PytorchSTFTFrameComputer.from_numpy_frame_computer`, which determines the
    below parameters from an :class:`STFTFrameComputer` which has already been
    initialized.

    The filters and window are learnable/adjustable. Be sure to disable gradients with
    :func:`torch.no_grad` if a fixed feature representation is desirable.

    Parameters
    ----------
    offsets_and_truncated_filters
        A sequence of pairs ``(offset, truncated_filter)``. `truncated_filter` is a
        one-dimensional tensor of the non-zero frequency response of a single filter in
        the bank. `offset` is the index in the short-time spectrum at which the
        `truncated_filter` begins.
    frame_length
        The number of audio samples constituting a frame.
    frame_shift
        The number of audio samples between subsequent frames.
    frame_style
        If ``'causal'``, the first frame begins at sample ``0``. Otherwise, the
        first frame is centered around sample ``0`` with the exact behaviour dictated
        by the `kaldi_shift` flag.
    window
        If specified, a tensor of shape ``(frame_length,)`` containing the windowing
        function. If unspecified, implicit rectangular windowing will be performed
        (with no gradient).
    dft_size
        The size of the spectrum to compute for each frame. Must be greater than
        or equal to `frame_length`. If unspecified, the first power of two at or beyond
        the frame length will be chosen.
    use_log
        Whether to take the logarithm of the resulting representation
    use_power
        Take the power spectrum instead of the magnitude spectrum
    include_energy
        Whether to add a coefficient at index 0 corresponding to the frame-wise energy
        of the signal
    kaldi_shift
        Dictates how to center frames when `frame_style` is :obj:`'centered'`. If
        :obj:`True`, the k-th frame will be computed using the signal between ``signal[
        k * frame_shift - frame_length // 2 + frame_shift // 2:k * frame_shift +
        (frame_length + 1) // 2 + frame_shift // 2]``. These are the frame bounds for
        Kaldi [povey2011]_. Otherwise, the k-th frame is ``signal[ k * frame_shift -
        (frame_length + 1) // 2 + 1: k * frame_shift + frame_length // 2 + 1]``.
    is_real
        Whether the filters are real in the time domain. If :obj:`True`, coefficients
        will be doubled (pre-log) to account for Hermitian symmetry.
    """

    __constants__ = (
        "centered",
        "dft_size",
        "frame_length",
        "frame_shift",
        "offsets",
        "include_energy",
        "use_log",
        "use_power",
    )

    centered: bool
    dft_size: int
    frame_length: int
    frame_shift: int
    offsets: Tuple[int, ...]
    include_energy: bool
    use_log: bool
    use_power: bool
    kaldi_shift: bool
    is_real: bool

    def __init__(
        self,
        offsets_and_truncated_filters: Sequence[Tuple[int, torch.Tensor]],
        frame_length: int,
        frame_shift: int,
        frame_style: Literal["centered", "causal"] = "centered",
        window: Optional[torch.Tensor] = None,
        dft_size: Optional[int] = None,
        use_log: bool = True,
        use_power: bool = False,
        include_energy: bool = False,
        kaldi_shift: bool = False,
        is_real: bool = False,
    ) -> None:
        offsets, filters = [], []
        for i, (offset, filter) in enumerate(offsets_and_truncated_filters):
            if filter.ndim != 1:
                raise ValueError(f"filter {i} is not a vector")
            elif not filter.size(0):
                raise ValueError(f"filter {i} is empty")
            check_positive(f"filter {i} offset", offset, True)
            offsets.append(offset)
            filters.append(filter)
        check_positive("frame_length", frame_length)
        check_positive("frame_shift", frame_shift)
        check_in("frame_style", frame_style, {"causal", "centered"})
        if window is not None:
            if window.shape != (frame_length,):
                raise ValueError(
                    f"Expected window.shape to be ({frame_length},); got {window.shape}"
                )
        if dft_size is None:
            dft_size = 2 ** math.ceil(math.log(frame_length, 2))
        elif dft_size < frame_length:
            raise ValueError(
                f"Expected dft_size to be gte {frame_length}; got {dft_size}"
            )
        super().__init__()
        self.frame_length, self.frame_shift = frame_length, frame_shift
        self.offsets, self.centered = tuple(offsets), frame_style == "centered"
        self.dft_size, self.use_log, self.use_power = dft_size, use_log, use_power
        self.kaldi_shift, self.is_real = kaldi_shift, is_real
        self.include_energy = include_energy

        self.filters = torch.nn.ParameterList(filters)
        if window is None:
            self.register_parameter("window", None)
        else:
            self.window = torch.nn.Parameter(window)

    @classmethod
    def from_stft_frame_computer(
        cls,
        computer: STFTFrameComputer,
        filter_type: torch.dtype = torch.cfloat,
        window_type: torch.dtype = torch.float,
    ) -> Self:
        """Create an instance using an STFTFrameComputer

        Parameters
        ----------
        computer
            The initialized instance to pull parameters from
        filter_type
            The data type to store filter parameters as
        window_type
            The data type to store the window parameter as
        """
        filters = [
            (o, torch.tensor(x, dtype=filter_type))
            for (o, x) in zip(computer._filt_start_idxs, computer._truncated_filts)
        ]
        frame_length = computer.frame_length
        frame_shift = computer.frame_shift
        frame_style = computer._frame_style
        window = torch.tensor(computer._window, dtype=window_type)
        dft_size = computer._dft_size
        use_log = computer._log
        use_power = computer._power
        include_energy = computer._include_energy
        kaldi_shift = computer._kaldi_shift
        is_real = computer._real
        return cls(
            filters,
            frame_length,
            frame_shift,
            frame_style,
            window,
            dft_size,
            use_log,
            use_power,
            include_energy,
            kaldi_shift,
            is_real,
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return pytorch_stft_frame_computer(
            signal,
            list(self.filters),
            self.offsets,
            self.frame_length,
            self.frame_shift,
            self.centered,
            self.window,
            self.dft_size,
            self.use_log,
            self.use_power,
            self.include_energy,
            self.kaldi_shift,
            self.is_real,
        )
