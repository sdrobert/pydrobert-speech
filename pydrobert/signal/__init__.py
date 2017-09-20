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
        found = []
        for subcls in chain(cls.__subclasses__()[::-1], [cls]):
            if alias in subcls.aliases:
                return subcls(*args, **kwargs)
        raise ValueError('Cannot find subclass with alias "{}"'.format(alias))

def alias_factory_subclass_from_arg(factory_class, arg):
    '''Boilerplate for getting an instance of an AliasedFactory

    Rather than an instance itself, a function could receive the
    arguments to initialize an AliasedFactory with ``from_alias``.
    This function uses the following strategy to try and do so::

    1. If ``arg`` is an instance of ``factory_class``, return ``arg``
    2. If ``arg`` is a string, use it as the alias
    3. a. Copy ``arg`` to a dictionary
       b. Pop the key ``'alias'`` and treat the rest as keyword arguments
       c. If the key ``'alias'`` is not found, try ``'name'``

    This function is intentionally limited in order to work nicely with
    JSON config files.
    '''
    if isinstance(arg, factory_class):
        return arg
    elif isinstance(arg, str):
        return factory_class.from_alias(arg)
    else:
        arg = dict(arg)
        try:
            alias = arg.pop('alias')
        except KeyError:
            alias = arg.pop('name')
        return factory_class.from_alias(alias, **arg)
