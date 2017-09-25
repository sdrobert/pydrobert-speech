"""Signal processing library, primarily for speech

Available modules
-----------------
compute
    Feature computations/transformations from signals
config
    Package constants
filters
    Filters and filter banks
post
    Post-processing of features
scales
    Scaling functions, including psychoacoustic scales such as Bark or
    Mel scales
util
    Miscellaneous functions for signal processing
vis
    Visualization functions. Requires `matplotlib` be installed
"""

import abc

from itertools import chain

from six import with_metaclass

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

__all__ = [
    'compute',
    'config',
    'filters',
    'post',
    'scales',
    'util',
]

class AliasedFactory(object, with_metaclass(abc.ABCMeta)):
    '''An abstract interface for initialing concrete subclasses with aliases'''

    aliases = set()

    @classmethod
    def from_alias(cls, alias, *args, **kwargs):
        '''Factory method for initializing a subclass that goes by an alias

        All subclasses of this class have the class attribute
        ``aliases``. This method matches ``alias`` to an element in some
        subclass' ``aliases`` and initializes it. Aliases of `cls` are
        included in the search. Alias conflicts are resolved by always
        trying to initialize the last registered subclass that matches
        the alias.

        Parameters
        ----------
        alias : str

        Raises
        ------
        ValueError
            Alias can't be found
        '''
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
