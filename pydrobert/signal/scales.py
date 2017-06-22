"""Scaling functions

Scaling functions transform a scalar in the frequency domain to some
other real domain (the "scale" domain). The scaling functions should be
invertible. Their primary purpose is to define the bandwidths of filters
in `pydrobert.signal.filters`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np

from six import with_metaclass

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

__all__ = [
    'ScalingFunction',
    'LinearScaling',
    'OctaveScaling',
    'MelScaling',
    'BarkScaling',
]

class ScalingFunction(object, with_metaclass(abc.ABCMeta)):
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

    def __init__(self, low_hz):
        if low_hz <= 0:
            raise ValueError('low_hz must be positive')
        self.low_hz = low_hz

    def scale_to_hertz(self, scale):
        return (2 ** scale) * max(1e-10, self.low_hz)

    def hertz_to_scale(self, hertz):
        return np.log2(hertz / max(1e-10, self.low_hz))

class MelScaling(ScalingFunction):
    r"""Psychoacoustic scaling function from [1]_

    Based of the experiment in [1]_ wherein participants adjusted a
    second tone until it was half the pitch of the first. The functional
    approximation to the scale is implemented with the formula from
    [2]_ (being honest, from Wikipedia):

    ..math:: s = 1127 \ln \left(1 + \frac{f}{700} \right)

    Where `s` is the scale and `f` is the frequency in Hertz.

    References
    ----------
    .. [1] S. S. Stevens, J. Volkmann & E. B. Newman (1937). A Scale for
       the Measurement of the Psychological Magnitude Pitch. The Journal
       of the Acoustical Society of America, 8, 185-190.
    .. [2] O'Shaughnessy, D. (1987). Speech communication: human and
       machine. Addison-Wesley Pub. Co.
    """

    def scale_to_hertz(self, scale):
        return 700. * (np.exp(scale / 1127.) - 1.)

    def hertz_to_scale(self, hertz):
        return 1127. * np.log(1 + hertz / 700.)

class BarkScaling(ScalingFunction):
    r"""Psychoacoustic scaling function from [1]_

    Based on a collection experiments briefly mentioned in [1]_
    involving masking to determine critical bands. The functional
    approximation to the scale is implemented with the formula from
    [2]_ (being honest, from Wikipedia):

    ..math::

         z = 26.81f/(1960 + f) - 0.53 \\
         s = \begin{cases}
            z + 0.15(2 - z) & \mbox{if }z < 2 \\
            z + 0.22(z - 20.1) & \mbox{if }z > 20.1
         \end{cases}

    Where `s` is the scale and `f` is the frequency in Hertz.

    References
    ----------
    .. [1] E. Zwicker (1961). Subdivision of the Audible Frequency Range
       into Critical Bands (Frequenzgruppen). The Journal of the
       Acoustical Society of America, 33, 248-248.
    .. [2] Hartmut Traunmuller (1990). Analytical expressions for the
       tonotopic sensory scale. The Journal of the Acoustical Society of
       America, 88, 97-100.
    """

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
