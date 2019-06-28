# Copyright 2019 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualization functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import cycle

import numpy as np

from pydrobert.speech.compute import LinearFilterBankFrameComputer

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"


def plot_frequency_response(
        banks, axes=None, dft_size=None, half=None, title=None,
        x_scale='hz', y_scale='dB', cmap=None):
    '''Plot frequency response of filters in a filter bank

    Parameters
    ----------
    bank : banks.LinearFilterBank or list
    axes : matplotlib.axes.Axes, optional
        An Axes object to plot on. Default is to generate a new figure
    dft_size : int, optional
        The size of the Discrete Fourier Transform to plot. Defaults to
        ``max(max(bank.supports), 2 * bank.sampling_rate /
        min(bank.supports_hz)``
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
    cmap : colormap, optinal
        A colormap to pull colours from. Defaults to matplotlib's default
        colormap

    Returns
    -------
    matplotlib.figure.Figure
        The containing figure`

    Raises
    ------
    ImportError
        If unable to import matplotlib
    '''
    from matplotlib import pyplot as plt
    from matplotlib import ticker
    try:
        len(banks)
    except AttributeError:  # 1 bank
        banks = [banks]
    if not all(x.num_filts for x in banks):
        raise ValueError(
            'Filter banks must have at least one filter to be visualized')
    if not all(x.sampling_rate == banks[0].sampling_rate for x in banks):
        raise ValueError(
            'Banks must all have the same sampling rate')
    rate = banks[0].sampling_rate
    if cmap is None:
        cmap = plt.get_cmap()
    if dft_size is None:
        dft_size = max(
            int(max(
                max(right - left for left, right in bank.supports),
                2 * rate / min(
                    right - left for left, right in bank.supports_hz),
            )) for bank in banks)
    if half is None:
        half = all(bank.is_real for bank in banks)
    if axes is None:
        fig, axes = plt.subplots()
    else:
        fig = axes.get_figure()
    colours = cmap.colors
    r_colours = list(cmap.colors)
    r_colours.reverse()
    responses_colours = []
    for bank, first_colour, second_color in zip(
            banks, cycle(colours), cycle(r_colours)):
        responses_colours.extend([
            (
                bank.get_frequency_response(filt_idx, dft_size, half=half),
                first_colour, second_color
            ) for filt_idx in range(bank.num_filts)
        ])
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
        axes.xaxis.set_major_locator(ticker.MultipleLocator(np.pi))
        axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        axes.xaxis.set_major_formatter(ticker.FuncFormatter(_pi_formatter))
        axes.xaxis.set_minor_formatter(ticker.FuncFormatter(_pi_formatter))
    elif x_scale == 'bins':
        x_title = 'DFT Bin'
    else:
        raise ValueError('Invalid x_scale: {}'.format(x_scale))
    if y_scale in ('db', 'dB', 'decibels'):
        y_title = 'Log Ratio (dB)'
        # maximum abs. Get ripped
        max_abs = max(
            max(np.abs(response)) for response, _, _ in responses_colours)
        max_abs = np.log10(max(np.finfo(float).eps, max_abs))
        for filt_idx in range(len(responses_colours)):
            response, first_colour, second_colour = responses_colours[filt_idx]
            response = np.abs(response)
            response[response <= np.finfo(float).eps] = np.nan
            response[...] = 20 * (np.log10(response) - max_abs)
            # looks better than discontinuities
            response[np.isnan(response)] = -1e10
            responses_colours[filt_idx] = response, first_colour, second_colour
        y_max = 0
        y_min = -10
    elif y_scale in ('pow', 'power'):
        y_title = 'Power'
        y_min = 0
        y_max = 0
        for filt_idx in range(len(responses_colours)):
            response, first_colour, second_colour = responses_colours[filt_idx]
            response = np.abs(response) ** 2
            y_max = max(y_max, max(response))
            responses_colours[filt_idx] = response, first_colour, second_colour
        y_max *= 1.04
    elif y_scale in ('real', 'imag', 'imaginary', 'both'):
        if y_scale == 'real':
            y_title = 'Real-value response'
        elif y_scale == 'both':
            y_title = 'Complex response'
        else:
            y_title = 'Imaginary-value response'
        y_min = np.inf
        y_max = -np.inf
        new_responses_colours = []
        for response, first_colour, second_colour in responses_colours:
            if y_scale == 'real':
                response = np.real(response)
            elif y_scale == 'both':
                response_b = np.imag(response)
                response = np.real(response)
                y_max = max(y_max, max(response_b))
                y_min = min(y_min, min(response_b))
                new_responses_colours.append(
                    (response_b, second_colour, first_colour))
            else:
                response = np.imag(response)
            y_max = max(y_max, max(response))
            y_min = min(y_min, min(response))
            new_responses_colours.append(
                (response, first_colour, second_colour))
        assert np.isfinite(y_min) and np.isfinite(y_max)
        y_max *= .96 if y_max < 0 else 1.04
        y_min *= .96 if y_min > 0 else 1.04
        del responses_colours
        responses_colours = new_responses_colours
    else:
        raise ValueError('Invalid y_scale: {}'.format(y_scale))
    axes.set_xlim((0, max(x)))
    axes.set_ylim((y_min, y_max))
    if title:
        axes.set_title(title)
    axes.set_ylabel(y_title)
    axes.set_xlabel(x_title)
    for response, colour, _ in responses_colours:
        axes.plot(x, response, color=colour)
    # if y_scale == 'both':
    #     real_handle = lines.Line2D([], [], color=real_colour, label='Real')
    #     imag_handle = lines.Line2D([], [], color=imag_colour, label='Imag')
    #     axes.legend(handles=[real_handle, imag_handle])
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


def compare_feature_frames(
        computers, signal, axes=None, figure_height=None, figure_width=None,
        plot_titles=None, positions=None, post_ops=None, title=None, **kwargs):
    '''Compare features from frame computers via spectrogram-like heat map

    Direct comparison of `FrameComputer` objects is possible because all
    subclasses of this abstract data type share a common interpretation
    of frame boundaries (according to `FrameComputer.frame_style`).

    Additional keyword args will be passed to the plotting routine.

    Parameters
    ----------
    computers : pydrobert.speech.compute.FrameComputer or tuple
        One or more frame computers to compare
    signal : array-like
        A 1D array of the raw speech. Assumed to be valid with respect
        to computer settings (e.g. sample rate).
    axes : matplotlib.axes.Axes or tuple, optional
        By default, this function creates a new figure and subplots.
        Setting one `axes` value for every `computers` value will plot
        feature representations from `computers` into each ordered Axes.
        If `axes` do not belong to the same figure, a `ValueError` will
        be raised
    figure_height : float, optional
        If a new figure is created, this sets the figure height (in
        inches). This value is determined dynamically according to
        `figure_width` by default. A `ValueError` will be raised if both
        `figure_height` and `axes` are set
    figure_width : float, optional
        If a new figure is created, this set the figure width (in
        inches). This value defaults to 3.33 inches if all subplots are
        positioned vertically, and to 7 inches if there are at least two
        columns of plots. A `ValueError` will be raised if both
        `figure_width` and `axes` are set
    plot_titles : tuple, optional
        An ordered list of strings specifying the titles of each
        subplot. The default is to not display subplot titles
    positions : tuple, optional
        If a new figure is created, `positions` decides how the
        subplots should be positioned relative to one another. Can
        contain only ints (describing the position on only the row-axis)
        or pairs of ints (describing the row-col positions). Positions
        must be contiguous and start from index 0 or 0,0 (top or
        top-left). `positions` cannot be specified if `axes` is
        specified
    post_ops : pydrobert.speech.post.PostProcessor or tuple, optional
        One or more post-processors to apply (in order) to each computed
        feature representation. If a simple list of post-processors is
        provided, each operation is applied to the default axis (the
        feature coefficient axis). To explicitly set the axis, pairs of
        ``(op, axis)`` can be specified in the list. No op is allowed
        to change the shape of the feature representation
        (e.g. `post.Deltas`), or a `ValueError` will be thrown
    title : str, optional
        The title of the whole figure. Default is to display no title

    Returns
    -------
    matplotlib.figure.Figure
        The containing figure

    Raises
    ------
    ImportError
        If unable to import matplotlib
    '''
    from matplotlib import pyplot as plt
    try:
        iter(computers)
    except TypeError:
        computers = (computers,)
    if not len(computers):
        raise ValueError('Expected at least one computer')
    if plot_titles is not None:
        try:
            iter(plot_titles)
        except TypeError:
            plot_titles = [plot_titles]
        if len(plot_titles) != len(computers):
            raise ValueError('Expected {} plot titles, got {}'.format(
                len(computers), len(plot_titles)))
    else:
        plot_titles = [None] * len(computers)
    if positions is not None:
        if len(computers) == 1 and positions not in (0, (0,), [0]):
            raise ValueError('Nonzero position specified for only one plot')
        elif axes is not None:
            raise ValueError('Cannot specify positions of predefined axes')
        elif len(positions) != len(computers):
            raise ValueError('Expected {} positions, got {}'.format(
                len(computers), len(positions)))
        if any(hasattr(p, '__iter__') for p in positions) and not \
                all(len(p) == 1 for p in positions if hasattr(p, '__iter__')):
            # expect 2-dimensional plot positioning
            if any(
                    not hasattr(p, '__iter__') or len(p) != 2
                    for p in positions):
                raise ValueError(
                    'Expected all plot positions to be two-dimensional')
            row_set = set(p[0] for p in positions)
            col_set = set(p[1] for p in positions)
            row_len, col_len = max(row_set) + 1, max(col_set) + 1
            if row_set != set(r for r in range(row_len)) or \
                    col_set != set(c for c in range(col_len)):
                raise ValueError('positions not contiguous')
            gs_args = (row_len, col_len)
        else:
            # expect 1-dimensional plot positioning. Using gridspec,
            # so have to add a column coordinate
            positions = tuple(
                (next(iter(p)), 0) if hasattr(p, '__iter__') else p
                for p in positions
            )
            row_set = set(p[0] for p in positions)
            row_len = max(row_set) + 1
            if row_set != set(r for r in range(row_len)):
                raise ValueError('positions not contiguous')
            gs_args = (row_len, 1)
    elif axes is None:
        # choose our own positions
        num_plots = len(computers)
        row_len = int(np.ceil(num_plots ** .5))
        col_len = row_len
        while col_len * row_len != num_plots:
            if col_len * row_len > num_plots and col_len > 1:
                row_len += 1
                col_len -= 1
            else:
                row_len -= 1
        gs_args = (row_len, col_len)
        positions = tuple(np.ndindex(gs_args))
    if figure_width is not None:
        if axes is not None:
            raise ValueError(
                'Cannot specify figure width with predefined axes')
    elif axes is None:
        figure_width = 7. if gs_args[1] > 1 else 3.33
    if figure_height is not None:
        if axes is not None:
            raise ValueError(
                'Cannot specify figure height with predefined axes')
    elif axes is None:
        figure_height = figure_width * 9 / 16 / gs_args[1] * gs_args[0]
    if post_ops is not None:
        try:
            iter(post_ops)
        except TypeError:
            post_ops = (post_ops,)
        if len(post_ops) == 2 and isinstance(post_ops[1], int):
            post_ops = (post_ops,)
    else:
        post_ops = []
    if axes is not None:
        try:
            iter(axes)
        except TypeError:
            axes = (axes,)
        if len(axes) != len(computers):
            raise ValueError('Expected {} axes, got {}'.format(
                len(computers), len(axes)))
        fig = axes[0].get_figure()
        for ax in axes[1:]:
            if ax.get_figure() != fig:
                raise ValueError('Axes do not share the same figure')
    else:
        fig = plt.figure(figsize=(figure_width, figure_height))
        if len(computers) == 1:
            axes = (fig.add_subplot(111),)
        else:
            axes = []
            sharey = all(
                isinstance(computer, LinearFilterBankFrameComputer)
                for computer in computers
            )
            gridspec = plt.GridSpec(gs_args[0], gs_args[1])
            for position in positions:
                if axes and sharey:
                    ax = fig.add_subplot(
                        gridspec[position], sharex=axes[0], sharey=axes[0])
                elif axes:
                    ax = fig.add_subplot(gridspec[position], sharex=axes[0])
                else:
                    ax = fig.add_subplot(gridspec[position])
                axes.append(ax)
    supremum_seconds = np.infty
    num_samples = len(signal)
    for idx, (computer, ax, plot_title) in enumerate(
            zip(computers, axes, plot_titles)):
        frame_length = computer.frame_length
        frame_shift = computer.frame_shift
        if computer.frame_style == 'causal':
            pad_left = 0
        else:  # centered
            pad_left = (frame_length + 1) // 2 - 1
        total_len = num_samples + pad_left
        num_frames = max(0, (total_len - frame_length) // frame_shift + 1)
        # individual computers may choose to add a final frame by
        # padding. Since this behaviour is not guaranteed, we only
        # consider full frames
        if not num_frames:
            raise ValueError(
                'The computer indexed at {} is unable to generate '
                'a full frame from the signal'.format(idx))
        # we use frame shifts to specify bounds (frame length is likely
        # overlapping), with the exception of the last frame
        sample_bounds = np.arange(num_frames + 1, dtype=float) * frame_shift
        if pad_left:
            # r.h.s. bound half a frame shift to right of center (or
            # half frame right of center for last frame)
            # l.h.s. bound half the other way (or 0 for first frame)
            sample_bounds[1:-1] -= (frame_shift + 1) // 2 - 1
            sample_bounds[-1] = sample_bounds[-2] + pad_left
        else:
            # l.h.s bound leftmost idx of each frame
            # r.h.s. is l.h.s. plus frame shift (or frame length for
            # last frame)
            sample_bounds[-1] = sample_bounds[-2] + frame_length
        seconds_bounds = sample_bounds / computer.sampling_rate
        supremum_seconds = min(supremum_seconds, seconds_bounds[-1])
        feat_slice = [slice(None, num_frames), slice(None)]
        if isinstance(computer, LinearFilterBankFrameComputer):
            yscale_label = 'Frequency (Hz)'
            bank = computer.bank
            num_coeffs = bank.num_filts
            if computer.includes_energy:
                feat_slice[-1] = slice(1, None)
            supports_hz = bank.supports_hz
            assert num_coeffs == len(supports_hz)
            centers_hz = tuple(
                (left + right) / 2 for left, right in supports_hz)
            # supports may be overlapping or sparse. Instead of using
            # supports to directly specify boundaries, we use them as
            # weights to pick points between center frequencies (except
            # the first and last filters, which get to extend their
            # lower and higher bounds to their supports, respectively.
            feature_bounds = np.empty(num_coeffs + 1)
            feature_bounds[0] = max(0, supports_hz[0][0])
            feature_bounds[-1] = min(
                computer.sampling_rate / 2, supports_hz[-1][-1])
            for high_idx in range(1, num_coeffs):
                low_c = centers_hz[high_idx - 1]
                high_c = centers_hz[high_idx]
                low_s, high_s = supports_hz[high_idx - 1]
                assert high_c >= low_c
                split_c = low_c * (high_s / (low_s + high_s))
                split_c += high_c * (low_s / (low_s + high_s))
                feature_bounds[high_idx] = split_c
        else:
            # no idea how to handle. Just plot rectangular coefficients
            num_coeffs = computer.num_coeffs
            yscale_label = None
            feature_bounds = np.arange(num_coeffs + 1)
        features = computer.compute_full(signal)
        assert features.shape[0] >= num_frames
        assert features[feat_slice].shape[-1] == num_coeffs
        for post_op_idx, post_op in enumerate(post_ops):
            try:
                apply_axis = post_op[1]
                post_op = post_op[0]
            except TypeError:
                apply_axis = -1
            new_features = post_op.apply(features, axis=apply_axis)
            if new_features.shape != features.shape:
                raise ValueError(
                    'The post_op indexed at {} changed the shape of the'
                    'features'.format(post_op_idx))
            features = new_features
        ax.pcolormesh(
            seconds_bounds, feature_bounds, features[feat_slice].T, **kwargs)
        if plot_title is not None:
            ax.set_title(plot_title)
        ax.set_xlabel('Time (seconds)')
        if yscale_label:
            ax.set_ylabel(yscale_label)
    for ax in axes:
        ax.set_xlim((0, supremum_seconds))
    if title:
        fig.suptitle(title)
    return fig
