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


"""Scaling functions

Scaling functions transform a scalar in the frequency domain to some
other real domain (the "scale" domain). The scaling functions should be
invertible. Their primary purpose is to define the bandwidths of filters
in `pydrobert.speech.filters`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np

from pydrobert.speech import AliasedFactory

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

__all__ = [
    'ScalingFunction',
    'LinearScaling',
    'OctaveScaling',
    'MelScaling',
    'BarkScaling',
]


class ScalingFunction(AliasedFactory):
    """Converts a frequency to some scale and back again"""

    @abc.abstractmethod
    def scale_to_hertz(self, scale):
        """Convert scale to frequency (in Hertz)"""
        pass

    @abc.abstractmethod
    def hertz_to_scale(self, hertz):
        """Convert frequency (in Hertz) to scalar"""
        pass


class LinearScaling(ScalingFunction):
    """Linear scaling between high and low scales/frequencies

    Parameters
    ----------
    low_hz : float
        The frequency (in Hertz) corresponding to scale 0.
    slope_hz : float, optional
        The increase in scale corresponding to a 1 Hertz increase in
        frequency

    Attributes
    ----------
    low_hz : float
    slope_hz : float
    """

    aliases = {'linear', 'uniform'}

    def __init__(self, low_hz, slope_hz=1.):
        self.low_hz = low_hz
        self.slope_hz = slope_hz

    def scale_to_hertz(self, scale):
        return scale / self.slope_hz + self.low_hz

    def hertz_to_scale(self, hertz):
        return (hertz - self.low_hz) * self.slope_hz


class OctaveScaling(ScalingFunction):
    """Uniform scaling in log2 domain from low frequency

    Parameters
    ----------
    low_hz : float
        The positive frequency (in Hertz) corresponding to scale 0.
        Frequencies below this value should never be queried.

    Attributes
    ----------
    low_hz : float

    Raises
    ------
    ValueError
        If `low_hz` is non-positive
    """

    aliases = {'octave'}

    def __init__(self, low_hz):
        if low_hz <= 0:
            raise ValueError('low_hz must be positive')
        self.low_hz = low_hz

    def scale_to_hertz(self, scale):
        return (2 ** scale) * max(1e-10, self.low_hz)

    def hertz_to_scale(self, hertz):
        return np.log2(hertz / max(1e-10, self.low_hz))


class MelScaling(ScalingFunction):
    r"""Psychoacoustic scaling function

    Based of the experiment in [stevens1937]_ wherein participants adjusted a
    second tone until it was half the pitch of the first. The functional
    approximation to the scale is implemented with the formula from
    [oshaughnessy1987]_ (being honest, from `Wikipedia
    <https://en.wikipedia.org/wiki/Mel_scale>`__):

    .. math:: s = 1127 \ln \left(1 + \frac{f}{700} \right)

    Where :math:`s` is the scale and :math:`f` is the frequency in Hertz.
    """

    aliases = {'mel'}

    def scale_to_hertz(self, scale):
        return 700. * (np.exp(scale / 1127.) - 1.)

    def hertz_to_scale(self, hertz):
        return 1127. * np.log(1 + hertz / 700.)


class BarkScaling(ScalingFunction):
    r"""Psychoacoustic scaling function

    Based on a collection experiments briefly mentioned in [zwicker1961]_
    involving masking to determine critical bands. The functional approximation
    to the scale is implemented with the formula from [tranmuller1990]_ (being
    honest, from `Wikipedia <https://en.wikipedia.org/wiki/Bark_scale>`__):

    .. math::

         s = \begin{cases}
            z + 0.15(2 - z) & \mbox{if }z < 2 \\
            z + 0.22(z - 20.1) & \mbox{if }z > 20.1
         \end{cases}

    where

    .. math:: z = 26.81f/(1960 + f) - 0.53

    Where :math:`s` is the scale and :math:`f` is the frequency in Hertz.
    """

    aliases = {'bark'}

    def scale_to_hertz(self, scale):
        bark = None
        if scale < 2:
            bark = (20. * scale - 6.) / 17.
        elif scale > 20.1:
            bark = (50. * scale + 221.1) / 61.
        else:
            bark = scale
        return 1960. * (bark + 0.53) / (26.28 - bark)

    def hertz_to_scale(self, hertz):
        bark = 26.81 * hertz / (1960. + hertz) - 0.53
        if bark < 2:
            return bark + 0.15 * (2. - bark)
        elif bark > 20.1:
            return bark + 0.22 * (bark - 20.1)
        else:
            return bark
