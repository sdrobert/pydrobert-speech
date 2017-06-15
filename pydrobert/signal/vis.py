"""Visualization methods for filters package

Requires the `matplotlib` package.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from matplotlib import lines
from matplotlib import pyplot as plt
from matplotlib import ticker

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

def plot_frequency_response(
        bank, axis=None, dft_size=None, half=None, title=None,
        x_scale='hz', y_scale='dB',
    ):
    '''Plot frequency response of filters in a filter bank

    Parameters
    ----------
    bank : banks.LinearFilterBank
    axis : matplotlib.axis.Axis, optional
        An axis object to plot on. Default is to generate a new figure
        and axis
    dft_size : int, optional
        The size of the Discrete Fourier Transform to plot. Defaults to
        `max(max(bank.supports), 2 * bank.sampling_rate /
        min(bank.supports_hz)`
    half : bool, optional
        Whether to plot the half or full spectrum. Defaults to
        ``bank.is_real``
    title : str, optional
        What to call the graph. The default is not to show a title
    x_scale : {'hz', 'ang', 'bins'}, optional
        The frequency coordinate scale along the x axis. Hertz
        (``'hz'``) is cycles/sec, angular frequency (``'ang'``) is
        radians/sec, and ``'bins'`` is the sample index within the DFT
    y_scale : {'dB', 'power', 'real', 'imag', 'both'}, optional
        How to express the frequency response along the y axis. Decibels
        (``'dB'``) is the log of a ratio of the maximum quantity in the
        bank. The range between 0 and -20 decibels is displayed. Power
        spectrum (``'power'``) is the squared magnitude of the frequency
        response. ``'real'`` is the real part of the response,
        ``'imag'`` is the imaginary part of the response, and ``'both'``
        displays both ``'real'`` and ``'imag'`` as separate lines

    Returns
    -------
    matplotlib.figure.Figure
    '''
    if not bank.num_filts:
        raise ValueError(
            'Filter bank must have at least one filter to be visualized')
    rate = bank.sampling_rate
    first_colour = 'b'
    second_colour = 'g'
    if dft_size is None:
        dft_size = max(max(bank.supports), 2 * rate / min(bank.supports_hz))
    if half is None:
        half = bank.is_real
    if axis is None:
        fig, axis = plt.subplots()
    else:
        fig = axis.get_figure()
    responses_colours = [
        (
            bank.get_frequency_response(filt_idx, dft_size, half=half),
            first_colour,
        )
        for filt_idx in range(bank.num_filts)
    ]
    if half:
        x = np.arange(
            (dft_size + dft_size % 2) // 2 + 1 - dft_size % 2,
            dtype=np.float32
        )
    else:
        x = np.arange(dft_size, dtype=np.float32)
    if x_scale in ('hz', 'Hz', 'hertz', 'Hertz'):
        x_title = 'Frequency (Hz)'
        x *= rate
        x /= dft_size
    elif x_scale in ('ang', 'angle', 'angular'):
        x_title = 'Angular Frequency'
        x *= 2 * np.pi
        x /= dft_size
        axis.xaxis.set_major_locator(ticker.MultipleLocator(np.pi))
        axis.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        axis.xaxis.set_major_formatter(ticker.FuncFormatter(_pi_formatter))
        axis.xaxis.set_minor_formatter(ticker.FuncFormatter(_pi_formatter))
    elif x_scale == 'bins':
        x_title = 'DFT Bin'
    else:
        raise ValueError('Invalid x_scale: {}'.format(x_scale))
    if y_scale in ('db', 'dB', 'decibels'):
        y_title = 'Log Ratio (dB)'
        # maximum abs. Get ripped
        max_abs = max(
            max(np.abs(response)) for response, _ in responses_colours)
        max_abs = np.log10(max(np.finfo(float).eps, max_abs))
        for filt_idx in range(len(responses_colours)):
            response, colour = responses_colours[filt_idx]
            response = np.abs(response)
            response[response <= np.finfo(float).eps] = np.nan
            response = 20 * (np.log10(response) - max_abs)
            # looks better than discontinuities
            response[np.isnan(response)] = -1e10
            responses_colours[filt_idx] = response, colour
        y_max = 0
        y_min = -20
    elif y_scale in ('pow', 'power'):
        y_title = 'Power'
        y_min = 0
        y_max = 0
        for filt_idx in range(len(responses_colours)):
            response, colour = responses_colours[filt_idx]
            response = np.abs(response) ** 2
            y_max = max(y_max, max(response))
        y_max *= 1.04
    elif y_scale in ('real', 'imag', 'imaginary', 'both'):
        if y_scale == 'real':
            y_title = 'Real-value response'
        elif y_scale == 'both':
            y_title = 'Complex response'
            real_colour = responses_colours[0][1]
            if real_colour == first_colour:
                imag_colour = second_colour
            else:
                imag_colour = first_colour
        else:
            y_title = 'Imaginary-value response'
        y_min = np.inf
        y_max = -np.inf
        new_responses_colours = []
        for response, colour in responses_colours:
            if y_scale == 'real':
                response = np.real(response)
            elif y_scale == 'both':
                colour = real_colour
                response_b = np.imag(response)
                response = np.real(response)
                y_max = max(y_max, max(response_b))
                y_min = min(y_min, min(response_b))
                new_responses_colours.append((response_b, imag_colour))
            else:
                response = np.imag(response)
            y_max = max(y_max, max(response))
            y_min = min(y_min, min(response))
            new_responses_colours.append((response, colour))
        assert np.isfinite(y_min) and np.isfinite(y_max)
        y_max *= .96 if y_max < 0 else 1.04
        y_min *= .96 if y_min > 0 else 1.04
        del responses_colours
        responses_colours = new_responses_colours
    else:
        raise ValueError('Invalid y_scale: {}'.format(y_scale))
    axis.set_xlim((0, max(x)))
    axis.set_ylim((y_min, y_max))
    if title:
        axis.set_title(title)
    axis.set_ylabel(y_title)
    axis.set_xlabel(x_title)
    for response, colour in responses_colours:
        axis.plot(x, response, color=colour)
    if y_scale == 'both':
        real_handle = lines.Line2D([], [], color=real_colour, label='Real')
        imag_handle = lines.Line2D([], [], color=imag_colour, label='Imag')
        axis.legend(handles=[real_handle, imag_handle])
    return fig

def _pi_formatter(val, _):
    num_halfpi = int(np.round(2 * val / np.pi))
    if np.isclose(num_halfpi * np.pi / 2, val):
        if not num_halfpi:
            return '0'
        elif num_halfpi == 1:
            return '\u03C0 / 2'
        elif num_halfpi == -1:
            return '-\u03C0 / 2'
        elif num_halfpi == 2:
            return '\u03C0'
        elif num_halfpi == -2:
            return '-\u03C0'
        elif num_halfpi % 2:
            return '{}\u03C0 / 2'.format(num_halfpi)
        else:
            return '{}\u03C0'.format(num_halfpi // 2)
    else:
        return ''

def compare_feature_frames(computers, signal, title=None, plot_titles=None):
    pass

# def compare_representations(banks, signal, cmvn=True, names=None):
#     '''Compare feature representations via spectrogram-like heat map

#     Parameters
#     ----------
#     banks : tuple or pydrobert.feats.FeatureBank
#         One or more `FeatureBank` objects
#     signal : str or 1D array-like
#         Either the samples of the signal to analyze or a path to a wave
#         file to analyze
#     cmvn : bool
#         Whether to perform Cepstral Mean-Variance Normalization per
#         feature coefficient
#     names : tuple, optional
#         The names of the respective banks passed. Default is to use
#         `str`

#     Returns
#     -------
#     matplotlib.figure.Figure
#     '''
#     if isinstance(banks, feats.FeatureBank):
#         banks = [banks]
#     if names is None:
#         names = [str(bank) for bank in banks]
#     sample_rate_hz = banks[0].sample_rate_hz
#     for bank in banks:
#         if bank.sample_rate_hz != sample_rate_hz:
#             raise ValueError('Banks do not have matching sample rates')
#     if isinstance(signal, str):
#         signal = _read_wave_file(signal, sample_rate_hz)
#     in_seconds = len(signal) / sample_rate_hz
#     sup_x, sup_y = np.infty, np.infty
#     inf_x, inf_y = -np.infty, -np.infty
#     fig, axes = plt.subplots(len(banks), sharex=True, sharey=True)
#     if len(banks) == 1:
#         axes = (axes,)
#     for bank, axis, name in zip(banks, axes, names):
#         frame_shift = bank.frame_shift
#         num_frames = int(len(signal) / frame_shift + .5)
#         frame_edges = np.arange(num_frames + 1, dtype='f') * frame_shift
#         frame_edges *= in_seconds / len(signal)
#         inf_x = max(frame_edges[0], inf_x)
#         sup_x = min(frame_edges[-1], sup_x)
#         num_coeffs = bank.num_coeffs
#         centers = bank.center_freqs_hz
#         assert num_coeffs == len(centers)
#         bws = bank.bandwidths_hz
#         assert num_coeffs == len(bws)
#         freq_edges = [centers[0] - bws[0] / 2]
#         for first_cent, first_bw, second_cent, second_bw in zip(
#                 centers[:-1], bws[:-1], centers[1:], bws[1:]):
#             edge = first_bw * first_cent + second_bw * second_cent
#             edge /= first_bw + second_bw
#             freq_edges.append(edge)
#         freq_edges.append(centers[-1] + bws[-1] / 2)
#         inf_y = max(freq_edges[0], inf_y)
#         sup_y = min(freq_edges[-1], sup_y)
#         feat_rep = bank.compute_full(signal)
#         if cmvn:
#             feat_rep = feat_rep - feat_rep.mean(0)
#             feat_rep = feat_rep / np.maximum(1e-10, feat_rep.std(0))
#         assert feat_rep.shape == (num_frames, num_coeffs)
#         axis.set_title(name)
#         axis.pcolormesh(frame_edges, freq_edges, feat_rep.T, cmap='plasma')
#     axes[0].set_xlim((inf_x, sup_x))
#     axes[0].set_ylim((inf_y, sup_y))
#     fig.text(0.92, 0.5, 'Frequency (Hz)', va='center', rotation=270)
#     axes[-1].set_xlabel('Time (sec)')
#     return fig

# def _read_wave_file(path, sample_rate):
#     buff = None
#     with wave.open(path) as wave_file:
#         if wave_file.getnchannels() != 1:
#             raise ValueError('"{}" is not mono'.format(path))
#         elif wave_file.getframerate() != sample_rate:
#             raise ValueError(
#                 '"{}" is not sampled at {}Hz'.format(path, sample_rate))
#         dtype_in = 'int{}'.format(8 * wave_file.getsampwidth())
#         buff = np.frombuffer(
#             wave_file.readframes(wave_file.getnframes()),
#             dtype=dtype_in
#         )
#         buff = buff.astype('float32')
#     return buff
