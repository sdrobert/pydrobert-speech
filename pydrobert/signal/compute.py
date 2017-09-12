"""Compute features from signals"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from itertools import count

import numpy as np

from six import with_metaclass

import pydrobert.signal as pysig

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

__all__ = [
    'FrameComputer',
    'LinearFilterBankFrameComputer',
    'ShortTimeFourierTransformFrameComputer',
    'ShortIntegrationFrameComputer',
    'frame_by_frame_calculation',
]


class FrameComputer(object, with_metaclass(abc.ABCMeta)):
    """Construct features from a signal from fixed-length segments

    A signal is treated as a (possibly overlapping) time series of
    frames. Each frame is transformed into a fixed-length vector of
    coefficients.

    Features can be computed one at a time, for example:
    >>> chunk_size = 2 ** 10
    >>> while len(signal):
    >>>     segment = signal[:chunk_size]
    >>>     feats = computer.compute_chunk(segment)
    >>>     # do something with feats
    >>>     signal = signal[chunk_size:]
    >>> feats = computer.finalize()

    Or all at once (which can be much faster, depending on how the
    computer is optimized):
    >>> feats = computer.compute_full(signal)

    The ``k``th frame can be roughly localized to the signal offset
    to about ``signal[k * computer.frame_shift]``. The signal's exact
    region of influence is dictated by the `frame_style` property.

    Attributes
    ----------
    frame_style : {'causal', 'centered'}
    sampling_rate : float
    frame_length : int
    frame_length_ms : float
    frame_shift : int
    frame_shift_ms : float
    num_coeffs : int
    started : bool
    """

    @property
    @abc.abstractmethod
    def frame_style(self):
        """Dictates how the signal is split into frames

        If ``'causal'``, the ``k``th frame is computed over the indices
        ``signal[k * frame_shift:k * frame_shift + frame_length]`` (at
        most). If ``'centered'``, the ``k``th frame is computed over
        the indices
        ``signal[
            k * frame_shift - (frame_length + 1) // 2 + 1:
            k * frame_shift + frame_length // 2 + 1]``. Any range
        beyond the bounds of the signal is generated in an
        implementation-specific way.
        """
        pass

    @property
    @abc.abstractmethod
    def sampling_rate(self):
        """Number of samples in a second of a target recording"""
        pass

    @property
    @abc.abstractmethod
    def frame_length(self):
        """Number of samples which dictate a feature vector

        .. warning:: Can be different from `get_next_segment_length`
        """
        pass

    @property
    def frame_length_ms(self):
        """Number of milliseconds of audio which dictate a feature vector"""
        return self.frame_length * 1000 / self.sampling_rate

    @property
    @abc.abstractmethod
    def frame_shift(self):
        """Number of samples absorbed between successive frame computations"""
        pass

    @property
    def frame_shift_ms(self):
        """Number of milliseconds between succecssive frame computations"""
        return self.frame_shift * 1000 / self.sampling_rate

    @property
    @abc.abstractmethod
    def num_coeffs(self):
        """Number of coefficients returned per frame"""
        pass

    @property
    @abc.abstractmethod
    def started(self):
        """Whether computations for a signal have started

        Becomes `True` after the first call to `compute_chunk`

        See also
        --------
        finalize : to return `started` to `False`
        """
        pass

    @abc.abstractmethod
    def compute_chunk(self, chunk):
        """Compute some coefficients, given a chunk of audio

        Parameters
        ----------
        chunk : array-like
            A 1D float array of the signal. Should be contiguous and
            non-overlapping with any previously processed segments in
            the audio stream

        Returns
        -------
        array-like
            A 2D float array of shape ``(num_frames, num_coeffs)``.
            ``num_frames`` is nonnegative (possibly 0). Contains some
            number of feature vectors, ordered in time over axis 0.
        """
        pass

    @abc.abstractmethod
    def finalize(self):
        """Conclude processing a stream of audio, processing any stored buffer

        Returns
        -------
        array-like
            A 2D float array of shape ``(num_frames, num_coeffs)``.
            ``num_frames`` is either 1 or 0.
        """
        pass

    def compute_full(self, signal):
        """Compute a full signal's worth of feature coefficients

        Parameters
        ----------
        signal : array-like
            A 1D float array of the entire signal

        Returns
        -------
        array-like
            A 2D float array of shape ``(num_frames, num_coeffs)``.
            ``num_frames`` is nonnegative (possibly 0). Contains some
            number of feature vectors, ordered in time over axis 0.

        Raises
        ------
        ValueError
            If already begin computing frames (``started=True``), and
            `reset` has not been called
        """
        return frame_by_frame_calculation(self, signal)

class LinearFilterBankFrameComputer(
        FrameComputer, with_metaclass(abc.ABCMeta)):
    '''Frame computers whose features are derived from linear filter banks

    Computers based on linear filter banks have a predictable number of
    coefficients and organization. Like the banks, the features with
    lower indices correspond to filters with lower bandwidths.
    `num_coeffs` will be simply `bank.num_filts + int(include_energy)`.

    Parameters
    ----------
    bank : LinearFilterBank
    include_energy : bool, optional
        Whether to include a coefficient based on the energy of the
        signal within the frame. If ``True``, the energy coefficient
        will be inserted at index 0.

    Attributes
    ----------
    bank : LinearFilterBank
    includes_energy : bool
    '''

    def __init__(self, bank, include_energy=False):
        self._bank = bank
        self._include_energy = bool(include_energy)

    @property
    def bank(self):
        '''The LinearFilterBank from which features are derived'''
        return self._bank

    @property
    def includes_energy(self):
        '''Whether the first coefficient is an energy coefficient'''
        return self._include_energy

    @property
    def num_coeffs(self):
        return self._bank.num_filts + int(self._include_energy)

class ShortTimeFourierTransformFrameComputer(LinearFilterBankFrameComputer):
    """Compute features of a signal by integrating STFTs

    Computations are per frame and as follows:
    1. The current frame is multiplied with some window (rectangular,
       Hamming, Hanning, etc)
    2. An DFT is performed on the result
    3. For each filter in the provided input bank:
       a. Multiply the result of 2. with the frequency response of the
          filter
       b. Sum either the pointwise square or absolute value of elements
          in the buffer from 3a.
       c. Optionally take the log of the sum

    .. warning:: This behaviour differs from that of [1]_ or [2]_ in
                 three ways. First, the sum (3b) comes after the
                 filtering (3a), which changes the result in the squared
                 case. Second, the sum is over the full power spectrum,
                 rather than just between 0 and the Nyquist. This
                 doubles the value at the end of 3c. if a real filter is
                 used. Third, frame boundaries are calculated
                 diffferently.

    Parameters
    ----------
    bank : LinearFilterBank
        Dictates the filters used to process the STFT. Each filter
        corresponds to one coefficient. `sampling_rate` is also dictated
        by this
    frame_length_ms : float, optional
        The length of a frame, in milliseconds. Defaults to the length
        of the largest filter in the bank
    frame_shift_ms : float, optional
        The offset between successive frames, in milliseconds
    frame_style : {'causal', 'centered'}, optional
        Defaults to ``'centered'`` if `bank.is_zero_phase`, ``'causal'``
        otherwise.
    include_energy : bool, optional
        Append an energy coefficient as the first coefficient of each
        frame
    pad_to_nearest_power_of_two : bool, optional
        Whether the DFT should be a padded to a power of two for
        computational efficiency
    window_name : {'rectangular', 'bartlett', 'blackman', 'hamming', 'hanning'}
        The name of the window used in step 1.
    use_log : bool, optional
        Whether to take the log of the sum from 3b.
    use_power : bool, optional
        Whether to sum the power spectrum or the magnitude spectrum

    Attributes
    ----------
    frame_style : {'causal', 'centered'}
    sampling_rate : float
    frame_length : int
    frame_length_ms : float
    frame_shift : int
    frame_shift_ms : float
    num_coeffs : int
    started : bool

    References
    ----------
    .. [1] Povey, D., et al (2011). The Kaldi Speech Recognition
           Toolkit. ASRU
    .. [2] Young, S. J., et al (2006). The HTK Book, version 3.4.
           Cambridge University Engineering Department
    """

    def __init__(
            self, bank, frame_length_ms=None, frame_shift_ms=10,
            frame_style=None, include_energy=False,
            pad_to_nearest_power_of_two=True,
            window_name='hanning', use_log=True, use_power=False):
        self._rate = bank.sampling_rate
        self._frame_shift = int(0.001 * frame_shift_ms * self._rate)
        self._log = use_log
        self._power = use_power
        self._real = bank.is_real
        self._started = False
        self._first_frame = True
        self._buf_len = 0
        self._chunk_dtype = np.float64
        if frame_style == None:
            frame_style = 'centered' if bank.is_zero_phase else 'causal'
        elif frame_style not in ('centered', 'causal'):
            raise ValueError('Invalid frame style: "{}"'.format(frame_style))
        self._frame_style = frame_style
        if frame_length_ms is None:
            self._frame_length = max(
                max(right - left for left, right in bank.supports),
                # ensure at least one dft bin is nonzero per filter
                int(np.ceil(2 * self._rate / min(
                    right - left for left, right in bank.supports_hz))),
            )
        else:
            self._frame_length = int(
                0.001 * frame_length_ms * bank.sampling_rate)
        self._buf = np.empty(self._frame_length, dtype=np.float64)
        if window_name == 'rectangular':
            self._window = np.ones(self._frame_length, dtype=float)
        elif window_name == 'bartlett' or window_name == 'triangular':
            self._window = np.bartlett(self._frame_length)
        elif window_name == 'blackman':
            self._window = np.blackman(self._frame_length)
        elif window_name == 'hamming':
            self._window = np.hamming(self._frame_length)
        elif window_name == 'hanning':
            self._window = np.hanning(self._frame_length)
        else:
            raise ValueError('Invalid window name: "{}"'.format(window_name))
        if pad_to_nearest_power_of_two:
            self._dft_size = int(2 ** np.ceil(np.log2(self._frame_length)))
        else:
            self._dft_size = self._frame_length
        if self._power:
            self._nonlin_op = lambda x: np.linalg.norm(x, ord=2) ** 2
        else:
            self._nonlin_op = lambda x: np.sum(np.abs(x))
        self._truncated_filts = []
        self._filt_start_idxs = []
        for filt_idx in range(bank.num_filts):
            start_idx, truncated_filt = bank.get_truncated_response(
                filt_idx, self._dft_size)
            self._filt_start_idxs.append(start_idx)
            self._truncated_filts.append(truncated_filt)
        super(ShortTimeFourierTransformFrameComputer, self).__init__(
            bank, include_energy=include_energy)

    @property
    def frame_style(self):
        return self._frame_style

    @property
    def sampling_rate(self):
        return self._rate

    @property
    def frame_length(self):
        return self._frame_length

    @property
    def frame_shift(self):
        return self._frame_shift

    @property
    def started(self):
        return self._started

    def _compute_frame(self, frame, coeffs):
        # given the frame, store feature values within coeff
        assert len(frame) == self._frame_length
        assert len(coeffs) == self.num_coeffs
        if self.includes_energy:
            coeffs[0] = np.inner(frame, frame)
            if not self._power:
                coeffs[0] **= .5
            if self._log:
                coeffs[0] = np.log(max(coeffs[0], pysig.LOG_FLOOR_VALUE))
            coeffs = coeffs[1:]
        if pysig.USE_FFTPACK:
            from scipy import fftpack
            is_odd = self._dft_size % 2
            buffered_frame = np.zeros(
                self._dft_size + 2 - is_odd, dtype=np.float64)
            buffered_frame[1:self._frame_length + 1] = frame
            buffered_frame[1:self._frame_length + 1] *= self._window
            buffered_frame[1:self._dft_size + 1] = fftpack.rfft(
                buffered_frame[1:self._dft_size + 1], overwrite_x=True)
            buffered_frame[0] = buffered_frame[1]
            buffered_frame[1] = 0
            half_spect = buffered_frame.view(np.complex128)
        else:
            half_spect = np.fft.rfft(frame * self._window, n=self._dft_size)
        assert half_spect.dtype == np.complex128
        half_len = len(half_spect)
        for filt_idx in range(len(self._filt_start_idxs)):
            start_idx = self._filt_start_idxs[filt_idx]
            truncated_filt = self._truncated_filts[filt_idx]
            trunc_len = len(truncated_filt)
            consumed = 0
            conjugate = False
            val = 0
            while consumed < trunc_len:
                if conjugate:
                    seg_len = min(
                        start_idx + trunc_len - consumed,
                        half_len - 2 + half_len % 2
                    ) - start_idx
                    seg_len = max(0, seg_len)
                    assert seg_len > 0
                    val += self._nonlin_op(
                        half_spect[
                            -2 + half_len % 2 - start_idx:
                            -2 + half_len % 2 - start_idx - seg_len:
                            -1
                        ].conj() * truncated_filt[consumed:consumed + seg_len]
                    )
                    start_idx -= half_len - 2 + half_len % 2
                else:
                    seg_len = min(start_idx + trunc_len - consumed, half_len)
                    seg_len -= start_idx
                    seg_len = max(0, seg_len)
                    if seg_len:
                        val += self._nonlin_op(
                            half_spect[
                                start_idx:start_idx + seg_len
                            ] * truncated_filt[consumed:consumed + seg_len]
                        )
                    start_idx -= half_len
                consumed += seg_len
                start_idx = max(0, start_idx)
            if self._real:
                val *= 2
            if self._log:
                val = np.log(max(val, pysig.LOG_FLOOR_VALUE))
            coeffs[filt_idx] = val

    def compute_chunk(self, chunk):
        self._chunk_dtype = chunk.dtype # needed for `finalize`
        # algorithm should work when frame shift is greater than frame
        # length - buf_len may be negative, which will skip samples
        buf_len = self._buf_len
        chunk_len = len(chunk)
        total_len = chunk_len + buf_len
        noncausal_first = self._frame_style == 'centered'
        noncausal_first &= self._first_frame
        if noncausal_first:
            frame_length = self._frame_length // 2 + 1
        else:
            frame_length = self._frame_length
        frame_shift = self._frame_shift
        num_frames = max(0, (total_len - frame_length) // frame_shift + 1)
        coeffs = np.empty(
            (num_frames, self.num_coeffs), dtype=self._chunk_dtype)
        for frame_idx in range(num_frames):
            frame_start_idx = frame_idx * frame_shift
            if frame_start_idx < buf_len:
                frame = np.concatenate([
                    self._buf[-(buf_len - frame_start_idx):],
                    chunk[:frame_length - buf_len + frame_start_idx],
                ])
            else:
                frame = chunk[
                    (frame_start_idx - buf_len):
                    (frame_start_idx - buf_len + frame_length)
                ]
            if noncausal_first:
                # the first frame's l.h.s is a reflection of its r.h.s.
                # shove the reflection into the buf - later frames
                # might need it
                chunk = chunk[(frame_length - buf_len):]
                chunk_len -= (frame_length - buf_len)
                frame_length = self._frame_length
                self._buf[:] = np.pad(
                    frame,
                    ((frame_length + 1) // 2 - 1, 0),
                    'reflect'
                )
                frame = self._buf
                total_len = chunk_len + frame_length
                buf_len = frame_length
                noncausal_first = False
            self._compute_frame(frame, coeffs[frame_idx])
            self._first_frame = False
        rem_len = total_len - num_frames * frame_shift
        assert rem_len < frame_length
        if rem_len > 0:
            throw_away = total_len - rem_len
            if throw_away < buf_len:
                rem_ring_len = buf_len - throw_away
                assert rem_ring_len < rem_len or \
                    (rem_ring_len <= rem_len and not len(chunk))
                self._buf[
                    self._frame_length - rem_len:
                    self._frame_length - rem_len + rem_ring_len
                ] = self._buf[self._frame_length - rem_ring_len:]
                self._buf[
                    self._frame_length - (rem_len - rem_ring_len):] = chunk
            else:
                self._buf[-rem_len:] = chunk[-rem_len:]
        self._buf_len = rem_len
        self._started = True
        return coeffs

    def finalize(self):
        buf_len = self._buf_len
        frame_length = self._frame_length
        if buf_len >= frame_length // 2 + 1:
            assert self._frame_style == 'causal' or not self._first_frame
            coeffs = np.empty((1, self.num_coeffs), dtype=self._chunk_dtype)
            frame = np.pad(
                self._buf[-buf_len:],
                (0, frame_length - buf_len),
                'reflect',
            )
            self._compute_frame(frame, coeffs[0])
        else:
            coeffs = np.empty((0, self.num_coeffs), dtype=self._chunk_dtype)
        self._buf_len = 0
        self._started = False
        self._first_frame = True
        return coeffs

    def compute_full(self, signal):
        if self.started:
            raise ValueError('Already started computing frames')
        # there should be a nicer way to calculate this
        frame_length = self._frame_length
        if len(signal) < frame_length // 2 + 1:
            return np.empty((0, self.num_coeffs), dtype=signal.dtype)
        if self._frame_style == 'causal':
            pad_left = 0
        else:
            pad_left = (self._frame_length + 1) // 2 - 1
        frame_shift = self._frame_shift
        total_len = pad_left + len(signal)
        num_frames = max(0, (total_len - frame_length) // frame_shift + 1)
        rem_len = total_len - num_frames * frame_shift
        if rem_len >= frame_length // 2 + 1:
            num_frames += 1
            pad_right = frame_length - rem_len
        else:
            pad_right = 0
        if pad_left or pad_right:
            signal = np.pad(signal, (pad_left, pad_right), 'reflect')
        coeffs = np.empty((num_frames, self.num_coeffs), dtype=signal.dtype)
        for frame_idx in range(num_frames):
            frame_left = frame_idx * frame_shift
            self._compute_frame(
                signal[frame_left:frame_left + frame_length],
                coeffs[frame_idx]
            )
        return coeffs

STFTFrameComputer = ShortTimeFourierTransformFrameComputer

class ShortIntegrationFrameComputer(LinearFilterBankFrameComputer):
    """Compute features by integrating over the filter modulus

    Each filter in the bank is convolved with the signal. A pointwise
    nonlinearity pushes the frequency band towards zero. Most of the
    energy of the signal can be captured in a short time integration.
    Though best suited to processing whole utterances at once, short
    integration is compatable with the frame analogy if the frame is
    assumed to be the cone of influence of the maximum-length filter.

    For computational purposes, each filter's impulse response is
    clamped to zero outside the support of the largest filter in the
    bank, making it a finite impulse response filter. This effectively
    decreases the frequency resolution of the filters which aren't
    already FIR. For better frequency resolution at the cost of
    computational time, increase `EFFECTIVE_SUPPORT_THRESHOLD`.

    Parameters
    ----------
    bank : LinearFilterBank
    frame_shift_ms : float, optional
        The offset between successive frames, in milliseconds. Also the
        length of the integration
    frame_style : {'causal', 'centered'}, optional
        Defaults to ``'centered'`` if `bank.is_zero_phase`, ``'causal'``
        otherwise. If ``'centered'`` each filter of the bank is
        translated so that its support lies in the center of the frame
    include_energy : bool, optional
        Append an energy coefficient as the first coefficient of each
        frame
    pad_to_nearest_power_of_two : bool, optional
        Pad the DFTs used in computation to a power of two for
        efficient computation
    use_power : bool, optional
        Whether the pointwise linearity is the signal's power or
        magnitude
    use_log : bool, optional
        Whether to take the log of the integration

    Attributes
    ----------
    bank : LinearFilterBank
    frame_style : {'causal', 'centered'}
    sampling_rate : float
    frame_length : int
    frame_length_ms : float
    frame_shift : int
    frame_shift_ms : float
    num_coeffs : int
    started : bool
    """

    def __init__(
            self, bank, frame_shift_ms=10, frame_style=None,
            include_energy=False, pad_to_nearest_power_of_two=True,
            use_power=False, use_log=True):
        self._rate = bank.sampling_rate
        self._frame_shift = int(.001 * frame_shift_ms * self._rate)
        self._log = bool(use_log)
        self._power = bool(use_power)
        self._real = bank.is_real
        self._ret_dtype = np.float64
        self._x_rem, self._y_rem, self._skip = 0, 0, 0
        self._started = False
        if frame_style == None:
            frame_style = 'centered' if bank.is_zero_phase else 'causal'
        elif frame_style not in ('centered', 'causal'):
            raise ValueError('Invalid frame style: "{}"'.format(frame_style))
        self._frame_style = frame_style
        if frame_style == 'centered':
            # we will recenter all filters so that their zero sample
            # is at max_support // 2
            self._max_support = max(
                right - left for left, right in bank.supports)
            self._translation = self._max_support // 2
        else:
            # we will shift all filters by whatever the minimum value
            # makes them all supported above/equal 0. We treat all that
            # translated space as nonzero for the sake of the
            # overlap-add algorithm
            self._translation = 0
            self._max_support = 0
            for left, right in bank.supports:
                self._translation = max(-left, self._translation)
                self._max_support = max(self._max_support, right)
            self._max_support += self._translation
        min_support_hz = min(right - left for left, right in bank.supports_hz)
        self._frame_length = self._max_support + self._frame_shift - 1
        self._dft_size = max(
            self._frame_length,
            # make sure the effective support is represented in at least
            # one dft bin
            int(np.ceil(2 * self._rate / min_support_hz)),
        )
        if pad_to_nearest_power_of_two:
            self._dft_size = int(2 ** np.ceil(np.log2(self._dft_size)))
        self._x_buf = np.empty(self._dft_size, dtype=np.float64)
        self._filts = []
        if include_energy:
            # this is a hacky way to make sure we get an accurate energy
            # coefficient - dirac deltas return the signal or a
            # translation of it
            dirac_filter = np.zeros(self._dft_size, dtype=np.float64)
            dirac_filter[self._translation] = 1
            if self._real:
                dirac_filter = np.fft.rfft(dirac_filter)
            else:
                dirac_filter = np.fft.fft(dirac_filter)
            self._filts.append(dirac_filter)
        for filt_idx in range(bank.num_filts):
            filt = bank.get_impulse_response(filt_idx, self._dft_size)
            if frame_style == 'centered':
                left_samp, right_samp = bank.supports[filt_idx]
                mid_samp = (left_samp + right_samp) // 2
                filt = np.roll(filt, self._translation - mid_samp + 1)
            else:
                filt = np.roll(filt, self._translation)
            # we clamp the support in time to make the filter FIR.
            self._filts.append(self._compute_dft(filt[:self._max_support]))
        # we don't have to store the filtered signal, just the values
        # that are accumulated in each frame shift
        y_blocks = self._dft_size - self._max_support + 2 * self._frame_shift
        y_blocks = int(np.ceil(y_blocks / self._frame_shift))
        self._y_buf = np.empty(
            (y_blocks, len(self._filts)),
            dtype=np.float64
        )
        super(ShortIntegrationFrameComputer, self).__init__(
            bank, include_energy=include_energy)

    @property
    def frame_style(self):
        return self._frame_style

    @property
    def sampling_rate(self):
        return self._rate

    @property
    def frame_length(self):
        return self._frame_length

    @property
    def frame_shift(self):
        return self._frame_shift

    @property
    def started(self):
        return self._started

    def compute_chunk(self, chunk):
        self._compute_preamble(chunk)
        chunk = self._handle_skip(chunk)
        chunk_len = len(chunk)
        valid_samples_per_dft = self._dft_size - self._max_support + 1
        num_raw = self._x_rem + chunk_len
        num_dfts = num_raw // valid_samples_per_dft
        num_frames = max(0, (num_raw + self._y_rem) // self._frame_shift - 1)
        if num_frames:
            num_processed = (num_frames + 1) * self._frame_shift
        else:
            num_processed = self._y_rem
        if num_processed - self._y_rem > num_dfts * valid_samples_per_dft:
            num_dfts += 1
        coeffs = np.empty((num_frames, self.num_coeffs), dtype=self._ret_dtype)
        cur_frame, chunk_copied = 0, 0
        for dft_idx in range(num_dfts):
            end_idx = min(
                (dft_idx + 1) * valid_samples_per_dft - self._x_rem,
                chunk_len
            )
            y_keep = end_idx - dft_idx * valid_samples_per_dft + self._x_rem
            start_idx = end_idx - self._dft_size
            if start_idx < 0:
                chunk_to_copy = end_idx - chunk_copied
                assert chunk_to_copy < self._dft_size
                self._x_buf[:self._dft_size - chunk_to_copy] = \
                    self._x_buf[chunk_to_copy:]
                self._x_buf[self._dft_size - chunk_to_copy:] = \
                    chunk[chunk_copied:end_idx]
                chunk_copied = end_idx
                cur_buf = self._x_buf
            else:
                cur_buf = chunk[start_idx:end_idx]
            X_buf = self._compute_dft(cur_buf)
            self._fill_y_buf(X_buf, y_keep)
            del X_buf
            while self._y_rem >= 2 * self._frame_shift:
                self._compute_frame(coeffs[cur_frame, :])
                cur_frame += 1
        assert cur_frame == num_frames, (cur_frame, num_frames)
        if chunk_len - chunk_copied:
            chunk_to_copy = min(self._dft_size, chunk_len - chunk_copied)
            self._x_buf[:-chunk_to_copy] = self._x_buf[chunk_to_copy:]
            self._x_buf[-chunk_to_copy:] = chunk[-chunk_to_copy:]
        self._x_rem = max(0, num_raw - num_dfts * valid_samples_per_dft)
        return coeffs

    def finalize(self):
        coeffs = np.empty((0, self.num_coeffs), dtype=self._ret_dtype)
        if self._started:
            # half a window of integration was provided at the start of
            # computations to y_rem so that integration remains
            # centered.
            if self._frame_style == 'centered':
                drop_center = self._frame_shift
            else:
                drop_center = 0
            remaining_samples = self._translation - self._skip + self._x_rem
            remaining_samples += self._y_rem - drop_center
            if remaining_samples >= self._frame_shift:
                coeffs = self.compute_chunk(np.zeros(
                    self._frame_shift + self._translation - drop_center,
                    dtype=self._ret_dtype))
        self._started = False
        return coeffs

    def compute_full(self, signal):
        if self._started:
            raise ValueError('Already started computing frames')
        return np.concatenate([
            self.compute_chunk(signal),
            self.finalize()
        ])

    def _compute_preamble(self, chunk):
        # check for data type consistency, handle stuff if just started
        if self._started:
            if chunk.dtype != self._ret_dtype:
                raise ValueError(
                    'Chunk does not share a type with previous chunks')
        else:
            if not np.issubdtype(chunk.dtype, np.float):
                raise ValueError('Chunk must be a float type')
            self._ret_dtype = chunk.dtype
            self._x_buf.fill(0)
            self._y_buf.fill(0)
            self._x_rem = 0
            if self._frame_style == 'centered':
                self._y_rem = self._frame_shift
            else:
                self._y_rem = 0
            self._skip = self._translation
            self._started = True

    def _handle_skip(self, chunk):
        # 'skip' refers to some number of samples at the beginning of
        # the utterance that won't count towards frames. We add them
        # to x_buf, but do not increment x_rem
        if not self._skip:
            return chunk
        assert not self._x_rem
        consumed = min(self._skip, len(chunk))
        x_len = len(self._x_buf)
        if consumed < x_len:
            self._x_buf[:x_len - consumed] = self._x_buf[consumed:]
            self._x_buf[x_len - consumed:] = chunk[:consumed]
        else:
            self._x_buf[:] = chunk[consumed - x_len:consumed]
        self._skip -= consumed
        return chunk[consumed:]

    def _fill_y_buf(self, X_buf, y_keep):
        # using the fourier domain of the raw signal window, compute
        # convolutions and store accumulations in y_buf
        block_offs = self._y_rem // self._frame_shift
        second_block_start = (block_offs + 1) * self._frame_shift - self._y_rem
        for filt_idx in range(self.num_coeffs):
            Y_buf = X_buf * self._filts[filt_idx]
            y_valid = self._compute_idft(Y_buf)[-y_keep:]
            del Y_buf
            for active_end, block_idx in zip(range(
                    second_block_start,
                    y_keep + self._frame_shift,
                    self._frame_shift), count(block_offs)):
                y_active = y_valid[
                    max(0, active_end - self._frame_shift):active_end]
                y_active /= self._frame_shift
                if self._power:
                    val = np.add.reduce((y_active * y_active.conj()).real)
                else:
                    val = np.add.reduce(abs(y_active))
                self._y_buf[block_idx, filt_idx] += val
        self._y_rem += y_keep

    def _compute_dft(self, buff):
        # given a buffer, compute its fourier transform. Always copies
        # the data
        assert len(buff) <= self._dft_size
        if pysig.USE_FFTPACK and self._real:
            from scipy import fftpack
            buffered_frame = np.zeros(
                self._dft_size + 2 - self._dft_size % 2, dtype=np.float64)
            buffered_frame[1:len(buff) + 1] = buff
            buffered_frame[1:self._dft_size + 1] = fftpack.rfft(
                buffered_frame[1:self._dft_size + 1], overwrite_x=True)
            buffered_frame[0] = buffered_frame[1]
            buffered_frame[1] = 0
            fourier_frame = buffered_frame.view(np.complex128)
        elif self._real:
            fourier_frame = np.fft.rfft(buff, n=self._dft_size)
        elif pysig.USE_FFTPACK:
            from scipy import fftpack
            complex_frame = np.zeros(self._dft_size, dtype=np.complex128)
            complex_frame[:len(buff)] = buff # implicit upcast if f32
            fourier_frame = fftpack.fft(complex_frame, overwrite_x=True)
        else:
            fourier_frame = np.fft.fft(buff, n=self._dft_size)
        assert fourier_frame.dtype == np.complex128
        return fourier_frame

    def _compute_idft(self, fourier_buff):
        # given a buffer, compute its inverse fourier transform. Assume
        # it's ok to modify the buffer.
        assert fourier_buff.dtype == np.complex128
        if pysig.USE_FFTPACK and self._real:
            from scipy import fftpack
            fourier_buff = fourier_buff.view(np.float64)
            fourier_buff[1] = fourier_buff[0]
            if self._dft_size % 2:
                fourier_buff = fourier_buff[1:]
            else:
                fourier_buff = fourier_buff[1:-1]
            idft = fftpack.irfft(fourier_buff, overwrite_x=True)
        elif self._real:
            idft = np.fft.irfft(fourier_buff, n=self._dft_size)
        elif pysig.USE_FFTPACK:
            from scipy import fftpack
            idft = fftpack.ifft(fourier_buff, overwrite_x=True)
        else:
            idft = np.fft.ifft(fourier_buff)
        return idft

    def _compute_frame(self, coeffs):
        # compute a frame's worth of coefficients from y_rem
        assert self._y_rem >= 2 * self._frame_shift
        coeffs[:] = np.add.reduce(self._y_buf[:2])
        if self._power:
            coeffs **= .5
        if self._log:
            coeffs[:] = np.log(np.maximum(coeffs, pysig.LOG_FLOOR_VALUE))
        self._y_buf[:-1] = self._y_buf[1:]
        self._y_buf[-1] = 0
        self._y_rem -= self._frame_shift

SIFrameComputer = ShortIntegrationFrameComputer

def frame_by_frame_calculation(computer, signal, chunk_size=2 ** 10):
    """Compute feature representation of entire signal iteratively

    This function constructs a feature matrix of a signal through
    successive calls to `computer.compute_chunk`. Its return value
    should be identical to that of calling
    `computer.compute_full(signal)`, but is possibly much slower.
    `computer.compute_full` should be favoured.

    Parameters
    ----------
    signal : array-like
        A 1D float array of the entire signal
    chunk_size : int
        The length of the signal buffer to process at a given time

    Returns
    -------
    array-like
        A 2D float array of shape ``(num_frames, num_coeffs)``.
        ``num_frames`` is nonnegative (possibly 0). Contains some number
        of feature vectors, ordered in time over axis 0.

    Raises
    ------
    ValueError
        If already begin computing frames (``computer.started ==
        True``)
    """
    if computer.started:
        raise ValueError('Already started computing frames')
    coeffs = []
    while len(signal):
        coeffs.append(computer.compute_chunk(signal[:chunk_size]))
        signal = signal[chunk_size:]
    coeffs.append(computer.finalize())
    return np.concatenate(coeffs)
