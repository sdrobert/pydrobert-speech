"""Compute features from signals"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np

from six import with_metaclass

from pydrobert.signal import LOG_FLOOR_VALUE
from pydrobert.signal import USE_FFTPACK

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

if USE_FFTPACK:
    from scipy import fftpack

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
            window_name='rectangular', use_log=True, use_power=False):
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
                max(bank.supports),
                # ensure at least one dft bin is nonzero per filter
                int(np.ceil(2 * self._rate / min(bank.supports_hz))),
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
                coeffs[0] = np.log(max(coeffs[0], LOG_FLOOR_VALUE))
            coeffs = coeffs[1:]
        if USE_FFTPACK:
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
                val = np.log(max(val, LOG_FLOOR_VALUE))
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
    energy of the signal can be captured in a short time integration
    (the frame shift). Though best suited to processing whole utterances
    at once, short integration is compatable with the frame analogy if
    the frame is assumed to be the cone of influence of the
    maximum-length filter.

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
        otherwise. If the bank uses zero-phase filters and `frame_style`
        is set to ``'causal'``, a phase of half of each filter's support
        is introduced to the filters so that they become causal. If the
        bank uses causal filters and `frame_style` is set to
        ``'centered'`` and the bank uses causal filters, the filters are
        converted to zero-phase filters with forward-backward filtering
        (squaring the frequency response).
    frequency_smoothing_pre : {'db2', 'tri3'}, optional
        Prior to integration but after the non-linearity, a low-pass
        convolution over filter channels can be performed to smooth out
        signal transients. ``'db2'`` is a 4-tap Daubechies scaling
        function which will push energy towards the higher-frequency
        filters. ``'tri3'`` is a 3-tap triangular filter that will keep
        the energy distribution centered, but blurred. The default
        is to do no frequency smoothing.
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
            frequency_smoothing_pre=None, include_energy=False,
            pad_to_nearest_power_of_two=True, use_power=False,
            use_log=True):
        self._rate = bank.sampling_rate
        self._frame_shift = int(0.001 * frame_shift_ms * self._rate)
        self._log = use_log
        self._power = use_power
        self._real = bank.is_real
        self._chunk_dtype = np.float64
        if frame_style == None:
            frame_style = 'centered' if bank.is_zero_phase else 'causal'
        elif frame_style not in ('centered', 'causal'):
            raise ValueError('Invalid frame style: "{}"'.format(frame_style))
        if frequency_smoothing_pre:
            if frequency_smoothing_pre == 'db2':
                self._freq_win = np.array(
                    [.6830127, 1.1830127, .3169873, -.1830127],
                    dtype=np.float64
                )
                self._freq_slice = slice(None, -3)
            elif frequency_smoothing_pre == 'tri3':
                self._freq_win = np.array([.33, .67, .33], dtype=np.float64)
                self._freq_slice = slice(1, -1)
            else:
                raise ValueError(
                    'Invalid frequency smoothing window: {}'.format(
                        frequency_smoothing_pre))
        else:
            self._freq_win = None
            self._feq_slice = None
        self._frame_style = frame_style
        self._max_support = max(bank.supports)
        if frame_style == 'centered':
            if not bank.is_zero_phase:
                self._max_support = 2 * self._max_support - 1
            self._skip = self._max_support // 2
        else:
            self._skip = 0
        if pad_to_nearest_power_of_two:
            self._frame_length = self._max_support + self._frame_shift - 1
            self._dft_size = int(2 ** np.ceil(np.log2(self._frame_length)))
            if self._dft_size < \
                    int(np.ceil(2 * self._rate / min(bank.supports_hz))):
                self._dft_size = int(2 ** np.ceil(np.log2(
                    self._rate / min(bank.supports_hz)) + 1))
        else:
            self._frame_length = max(
                self._max_support + 1,
                int(np.ceil(2 * self._rate / min(bank.supports_hz)))
            ) + self._frame_shift - 1
            self._dft_size = self._frame_length
        self._filts = []
        if include_energy:
            # this is a hacky way to make sure we get an accurate energy
            # coefficient - dirac deltas return the signal or a
            # translation of it
            dirac_filter = np.zeros(self._dft_size, dtype=np.float64)
            dirac_filter[self._skip] = 1
            if self._real:
                dirac_filter = np.fft.rfft(dirac_filter)
            else:
                dirac_filter = np.fft.fft(dirac_filter)
            self._filts.append(dirac_filter)
        for filt_idx in range(bank.num_filts):
            if frame_style == 'centered':
                if bank.is_zero_phase:
                    filt = bank.get_impulse_response(filt_idx, self._dft_size)
                else:
                    # make zero-phase with forward-backward
                    filt = bank.get_frequency_response(
                        filt_idx, self._dft_size, half=self._real)
                    filt = np.abs(filt) ** 2
                    filt = self._compute_idft(filt)
                filt = np.roll(filt, self._max_support // 2)
                self._skip = self._max_support // 2
            else: # causal
                filt = bank.get_impulse_response(filt_idx, self._dft_size)
                if bank.is_zero_phase:
                    support = bank.supports[filt_idx]
                    filt = np.roll(filt, support // 2)
                self._skip = 0
            # we clamp the support in time to make the filter FIR.
            self._filts.append(self._compute_dft(filt[:self._max_support]))
        self._jump = self._dft_size - self._max_support + 1
        assert self._jump > 0
        self._x_len = self._dft_size + self._frame_shift - 1
        self._x_buf = np.zeros(self._x_len)
        self._x_rem = 0
        self._started = False
        self._first_frame = True
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

    def _compute_dft(self, buff):
        # given a buffer, compute its fourier transform. Always copies
        # the data
        assert len(buff) <= self._dft_size
        if USE_FFTPACK and self._real:
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
        elif USE_FFTPACK:
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
        if USE_FFTPACK and self._real:
            fourier_buff = fourier_buff.view(np.float64)
            fourier_buff[1] = fourier_buff[0]
            if self._dft_size % 2:
                fourier_buff = fourier_buff[1:]
            else:
                fourier_buff = fourier_buff[1:-1]
            idft = fftpack.irfft(fourier_buff, overwrite_x=True)
        elif self._real:
            idft = np.fft.irfft(fourier_buff, n=self._dft_size)
        elif USE_FFTPACK:
            idft = fftpack.ifft(fourier_buff, overwrite_x=True)
        else:
            idft = np.fft.ifft(fourier_buff)
        return idft

    def _compute_frame(self, y_buf, coeffs):
        # given a frame shift's worth of convolved coefficients and a
        # vector to put coefficients, compute them
        if self._freq_win is not None:
            samples = y_buf[int(self.includes_energy):]
            samples = np.apply_along_axis(
                lambda col: np.convolve(
                    col, self._freq_win)[self._freq_slice],
                0,
                samples,
            )
            y_buf[int(self.includes_energy):] = samples
            del samples
        for coeff_idx in range(self.num_coeffs):
            if self._power:
                val = np.linalg.norm(y_buf[coeff_idx], ord=2) ** 2
            else:
                val = np.sum(np.abs(y_buf[coeff_idx]))
            if self._log:
                val = np.log(max(val, LOG_FLOOR_VALUE))
            coeffs[coeff_idx] = val

    def compute_chunk(self, chunk):
        self._chunk_dtype = chunk.dtype
        # on entry:
        #  x_buf contains the past dft_size samples. x_rem is the
        #  number of which haven't been convolved yet (r.h.s. of x_buf).
        chunk_len = len(chunk)
        if self._skip and chunk_len:
            assert not self._x_rem
            consumed = min(self._skip, chunk_len)
            if consumed < self._x_len:
                self._x_buf[:self._x_len - consumed] = self._x_buf[consumed:]
                self._x_buf[self._x_len - consumed:] = chunk[:consumed]
            else:
                self._x_buf[:] = chunk[consumed - self._x_len:consumed]
            self._skip -= consumed
            chunk_len -= consumed
            chunk = chunk[consumed:]
        total_samples = chunk_len + self._x_rem
        centered_first = self._frame_style == 'centered' and self._first_frame
        if centered_first:
            frame_shift = self._frame_shift // 2 + 1
        else:
            frame_shift = self._frame_shift
        num_frames = (total_samples - frame_shift) // self._frame_shift + 1
        y_len = max(0, (num_frames - 1) * self._frame_shift + frame_shift)
        coeffs = np.empty((num_frames, self.num_coeffs), dtype=chunk.dtype)
        num_dfts = int(np.ceil(y_len / self._jump))
        y_buf = np.empty(
            (self.num_coeffs, self._jump + self._frame_shift - 1),
            dtype=np.float64 if self._real else np.complex128,
        )
        frame_idx = 0
        unprocessed = 0
        for dft_idx in range(num_dfts):
            end_idx = min((dft_idx + 1) * self._jump, y_len)
            num_new = end_idx - dft_idx * self._jump
            # start_idx can be very negative for the first few frames.
            # this just means we're using more of our history buffer
            start_idx = end_idx - self._dft_size
            if start_idx < self._x_rem:
                x_k = np.concatenate([
                    self._x_buf[
                        start_idx - self._x_rem:
                        min(end_idx - self._x_rem, 0) + self._x_len
                    ],
                    chunk[:max(0, end_idx - self._x_rem)]
                ])
            else:
                x_k = chunk[start_idx - self._x_rem:end_idx - self._x_rem]
            assert len(x_k) == self._dft_size
            X_k = self._compute_dft(x_k)
            for filt_idx in range(self.num_coeffs):
                Y_k = X_k * self._filts[filt_idx]
                y_k = self._compute_idft(Y_k)
                y_buf[filt_idx, unprocessed:unprocessed + num_new] = \
                    y_k[-num_new:]
            unprocessed += num_new
            while unprocessed >= frame_shift:
                self._compute_frame(y_buf[:, :frame_shift], coeffs[frame_idx])
                frame_idx += 1
                unprocessed -= frame_shift
                y_buf[
                    :, :unprocessed
                ] = y_buf[:, frame_shift:frame_shift + unprocessed]
                self._first_frame = False
                frame_shift = self._frame_shift
                centered_first = False
        assert unprocessed == 0 and frame_idx == num_frames
        self._x_rem = total_samples - y_len
        if chunk_len >= self._x_len:
            self._x_buf[:] = chunk[-self._x_len:]
        elif chunk_len > 0:
            self._x_buf[:self._x_len - chunk_len] = self._x_buf[chunk_len:]
            self._x_buf[self._x_len - chunk_len:] = chunk
        self._started = True
        return coeffs

    def finalize(self):
        if self._frame_style == 'centered':
            coeffs = self.compute_chunk(np.zeros(
                self._max_support // 2,
                dtype=self._chunk_dtype,
            ))
        else:
            coeffs = np.empty((0, self.num_coeffs), dtype=self._chunk_dtype)
        if self._x_rem >= self._frame_shift // 2 + 1:
            coeffs_2 = self.compute_chunk(np.zeros(
                self._frame_shift - self._x_rem,
                dtype=self._chunk_dtype,
            ))
            assert coeffs_2.shape[0] == 1 and not self._x_rem
            coeffs = np.concatenate([coeffs, coeffs_2])
        self._x_rem = 0
        self._x_buf.fill(0)
        self._started = False
        self._first_frame = True
        if self._frame_style == 'centered':
            self._skip = self._max_support // 2
        else:
            self._skip = 0
        return coeffs

    def compute_full(self, signal):
        if self.started:
            raise ValueError('Already started computing frames')
        if self._frame_style == 'centered':
            offset = self._max_support // 2
            first_frame_shift = self._frame_shift // 2 + 1
        else:
            offset = 0
            first_frame_shift = self._frame_shift
        signal_len = len(signal)
        num_frames = (signal_len - first_frame_shift) // self._frame_shift + 1
        y_len = max(
            0, (num_frames - 1) * self._frame_shift + first_frame_shift)
        if signal_len - y_len >= self._frame_shift // 2 + 1:
            num_frames += 1
            y_len += self._frame_shift
        num_dfts = int(np.ceil(y_len / self._jump))
        coeffs = np.empty((num_frames, self.num_coeffs), dtype=signal.dtype)
        y_buf = np.empty(
            (self.num_coeffs, self._jump + self._frame_shift - 1),
            dtype=np.float64 if self._real else np.complex128,
        )
        unprocessed = 0
        frame_shift = first_frame_shift
        frame_idx = 0
        for dft_idx in range(num_dfts):
            end_idx = min((dft_idx + 1) * self._jump, y_len) + offset
            start_idx = end_idx - self._dft_size
            num_new = end_idx - dft_idx * self._jump - offset
            if start_idx < 0 or end_idx > signal_len:
                x_k = np.pad(
                    signal[max(0, start_idx):end_idx],
                    (
                        max(0, -start_idx),
                        max(0, end_idx - max(start_idx, signal_len)),
                    ),
                    'constant',
                )
            else:
                x_k = signal[start_idx:end_idx]
            assert len(x_k) == self._dft_size
            X_k = self._compute_dft(x_k)
            for filt_idx in range(self.num_coeffs):
                Y_k = X_k * self._filts[filt_idx]
                y_k = self._compute_idft(Y_k)
                y_buf[filt_idx, unprocessed:unprocessed + num_new] = \
                    y_k[-num_new:]
            unprocessed += num_new
            while unprocessed >= frame_shift:
                self._compute_frame(y_buf[:, :frame_shift], coeffs[frame_idx])
                frame_idx += 1
                unprocessed -= frame_shift
                y_buf[
                    :, :unprocessed
                ] = y_buf[:, frame_shift:frame_shift + unprocessed]
                frame_shift = self._frame_shift
        assert not unprocessed and frame_idx == num_frames
        return coeffs

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
