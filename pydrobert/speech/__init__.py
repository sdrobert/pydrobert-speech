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

"""Speech processing library

References
----------
.. [stevens1937] S. S. Stevens, J. Volkmann, and E. B. Newman, "A Scale for the
   Measurement of the Psychological Magnitude Pitch," The Journal of the
   Acoustical Society of America, vol. 8, no. 3, pp. 185-190, 1937.
.. [flanagan1960] J. L. Flanagan, "Models for approximating basilar membrane
   displacement," The Bell System Technical Journal, vol. 39, no. 5, pp.
   1163-1191, Sep. 1960.
.. [zwicker1961] E. Zwicker, "Subdivision of the Audible Frequency Range into
   Critical Bands (Frequenzgruppen)," The Journal of the Acoustical Society of
   America, vol. 33, no. 2, pp. 248-248, 1961.
.. [aertsen1981] A. M. H. J. Aertsen, J. H. J. Olders, and P. I. M. Johannesma,
   "Spectro-temporal receptive fields of auditory neurons in the grassfrog,"
   Biological Cybernetics, vol. 39, no. 3, pp. 195-209, Jan. 1981.
.. [oshaughnessy1987] D. O'Shaughnessy, Speech communication: human and
   machine. Addison-Wesley Pub. Co., 1987.
.. [tranmuller1990] H. Traunm\\:{u}ller, "Analytical expressions for the
   tonotopic sensory scale," The Journal of the Acoustical Society of America,
   vol. 88, no. 1, pp. 97-100, Jul. 1990.
.. [povey2011] D. Povey et al., "The Kaldi Speech Recognition Toolkit," in
   IEEE 2011 Workshop on Automatic Speech Recognition and Understanding, Hilton
   Waikoloa Village, Big Island, Hawaii, US, 2011.
.. [young] S. Young et al., "The HTK book (for HTK version 3.4)," Cambridge
   university engineering department, vol. 2, no. 2, pp. 2-3, 2006.
"""

import abc

from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

from future.utils import with_metaclass

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"

__all__ = [
    'AliasedFactory',
    'compute',
    'config',
    'filters',
    'post',
    'pre',
    'scales',
    'util',
    'vis',
]

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = 'dev'


class AliasedFactory(object, with_metaclass(abc.ABCMeta)):
    '''An abstract interface for initialing concrete subclasses with aliases'''

    aliases = set()

    @classmethod
    def from_alias(cls, alias, *args, **kwargs):
        '''Factory method for initializing a subclass that goes by an alias

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
