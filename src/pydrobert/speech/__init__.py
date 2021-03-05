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

import abc

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2021 Sean Robertson"

__all__ = [
    "AliasedFactory",
    "compute",
    "config",
    "filters",
    "post",
    "pre",
    "scales",
    "util",
    "vis",
]

try:
    from .version import version as __version__  # type: ignore
except ImportError:
    __version__ = "inplace"


class AliasedFactory(abc.ABC):
    """An abstract interface for initialing concrete subclasses with aliases"""

    aliases = set()

    @classmethod
    def from_alias(cls, alias: str, *args, **kwargs):
        """Factory method for initializing a subclass that goes by an alias

        All subclasses of this class have the class attribute ``aliases``. This
        method matches `alias` to an element in some subclass' ``aliases`` and
        initializes it. Aliases of this class are included in the search. Alias
        conflicts are resolved by always trying to initialize the last
        registered subclass that matches the alias.

        Parameters
        ----------
        alias : str

        Raises
        ------
        ValueError
            Alias can't be found
        """
        stack = [cls]
        pushed_children = set()
        while stack:
            parent = stack.pop()
            if parent not in pushed_children:
                children = parent.__subclasses__()
                stack.append(parent)
                stack.extend(children)
                pushed_children.add(parent)
            elif alias in parent.aliases:
                return parent(*args, **kwargs)
        raise ValueError('Cannot find subclass with alias "{}"'.format(alias))
