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

'''Package constants used throughout pydrobert.speech'''

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

USE_FFTPACK = False
"""
The scipy implementation of the FFT can be much faster than the numpy
one. This is set automatically to ``True`` if `scipy.fftpack` can be
imported. It can be set to ``False`` to use the numpy implementation.
"""
try:
    from scipy import fftpack
    USE_FFTPACK = True
except ImportError:
    pass

EFFECTIVE_SUPPORT_THRESHOLD = 5e-4
"""
No function is compactly supported in both the time and Fourier domains,
but large regions of either domain can be very close to zero. This
value serves as a threshold for zero. The higher it is, the more
accurate computations will be, but the longer they will take
"""

LOG_FLOOR_VALUE = 1e-5
"""Value used as floor when taking log in computations"""
