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

"""Speech processing library"""

import warnings

from .alias import AliasedFactory as _AF

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2021 Sean Robertson"

__all__ = [
    "alias",
    "compute",
    "config",
    "corpus",
    "filters",
    "post",
    "pre",
    "scales",
    "util",
]


class AliasedFactory(_AF):
    @classmethod
    def from_alias(cls, alias: str, *args, **kwargs):
        warnings.warn(
            "using AliasedFactory from pydrobert.speech is deprecated. Import from "
            "pydrobert.speech.alias",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return super().from_alias(alias, *args, **kwargs)


try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    __version__ = "inplace"

