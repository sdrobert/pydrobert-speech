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

"""Package constants used throughout pydrobert.speech"""

from typing import Set

__all__ = [
    "EFFECTIVE_SUPPORT_THRESHOLD",
    "LOG_FLOOR_VALUE",
    "SOUNDFILE_SUPPORTED_FILE_TYPES",
    "USE_FFTPACK",
]


USE_FFTPACK = False
"""bool : Whether to use :mod:`scipy.fftpack`

The scipy implementation of the FFT can be much faster than the numpy one. This is set
automatically to :obj:`True` if :mod:`scipy.fftpack` can be imported. It can be set to
:obj:`False` to use the numpy implementation.

:meta hide-value:
"""
try:
    from scipy import fftpack  # noqa: F401

    USE_FFTPACK = True
except ImportError:
    pass

EFFECTIVE_SUPPORT_THRESHOLD = 5e-4
"""float : Value considered roughly zero for support computations

No function is compactly supported in both the time and Fourier domains, but large
regions of either domain can be very close to zero. This value serves as a threshold for
zero. The higher it is, the more accurate computations will be, but the longer they will
take
"""

LOG_FLOOR_VALUE = 1e-5
"""float : Value used as floor when taking log in computations"""


# N.B. libsndfile's sphere support currently can't decode as many sphere encodings
# as _sphere.py can
_BASE_SOUNDFILE_SUPPORTED_TYPES = {"wav", "ogg", "flac", "aiff"}
_FULL_SOUNDFILE_SUPPORTED_TYPES = set()

SOUNDFILE_SUPPORTED_FILE_TYPES: Set[str] = set()
"""set : A list of the types of files SoundFile will be responsible for reading

If :mod:`soundfile` can be imported, it's the intersection of
:func:`soundfile.available_formats` with the set "wav", "ogg", "flac", and "aiff".

:meta hide-value:

See Also
--------
pydrobert.speech.util.read_signal
    Where this flag is used
"""

try:
    import soundfile as sf

    _FULL_SOUNDFILE_SUPPORTED_TYPES = set(x.lower() for x in sf.available_formats())

    SOUNDFILE_SUPPORTED_FILE_TYPES = (
        _BASE_SOUNDFILE_SUPPORTED_TYPES & _FULL_SOUNDFILE_SUPPORTED_TYPES
    )

except ImportError:
    pass

