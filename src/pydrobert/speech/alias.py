# Copyright 2022 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functionality to do with alias factories"""

import abc
from typing import Any, Mapping, Set, Type, TypeVar, Union, Type

__all__ = [
    "alias_factory_subclass_from_arg",
    "AliasedFactory",
]

T = TypeVar("T", bound="AliasedFactory", covariant=True)


class AliasedFactory(abc.ABC):
    """An abstract interface for initialing concrete subclasses with aliases"""

    aliases: Set[str] = set()
    """class aliases for :func:`from_alias`"""

    @classmethod
    def from_alias(cls: Type[T], alias: str, *args, **kwargs) -> T:
        """Factory method for initializing a subclass that goes by an alias

        All subclasses of this class have the class attribute `aliases`. This method
        matches `alias` to an element in some subclass' `aliases` and initializes it.
        Aliases of this class are included in the search. Alias conflicts are resolved
        by always trying to initialize the last registered subclass that matches the
        alias.

        Parameters
        ----------
        alias
            Alias of the subclass
        *args
            Positional arguments to initialize the subclass
        **kwargs
            Keyword arguments to initialize the subclass

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
        raise ValueError(f"Cannot find subclass with alias '{alias}'")


def alias_factory_subclass_from_arg(
    factory_class: Type[T], arg: Union[T, str, Mapping[str, Any]]
) -> T:
    """Boilerplate for getting an instance of an AliasedFactory

    Rather than an instance itself, a function could receive the arguments to initialize
    an :class:`AliasedFactory` with :func:`AliasedFactory.from_alias`. This function
    uses the following strategy to try and do so

    1. If `arg` is an instance of `factory_class`, return `arg`
    2. If `arg` is a :class:`str`, use it as the alias
    3. a. Copy `arg` to a dictionary
       b. Pop the key :obj:`'alias'` and treat the rest as keyword arguments
       c. If the key :obj:`'alias'` is not found, try :obj:`'name'`

    This function is intentionally limited in order to work nicely with JSON config
    files.
    """
    if isinstance(arg, factory_class):
        return arg
    elif isinstance(arg, str):
        return factory_class.from_alias(arg)
    else:
        arg = dict(arg)
        try:
            alias = arg.pop("alias")
        except KeyError:
            alias = arg.pop("name")
        return factory_class.from_alias(alias, **arg)
