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

'''Submodule for corpus iterators'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import cycle

from pydrobert.speech.post import PostProcessor
from pydrobert.speech.util import alias_factory_subclass_from_arg


__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    'post_process_wrapper'
]


def post_process_wrapper(cls):
    '''Wrap a pydrobert-kaldi Data object for post-processing

    This function returns a class that wraps the `cls` class, performing
    some post-processing after batching
    '''
    class _Wrapper(cls):
        def __init__(self, table, *additional_tables, **kwargs):
            postprocessors = kwargs.pop('postprocessors', dict())
            if not hasattr(postprocessors, 'get'):
                postprocessors = {0: postprocessors}
            for key, value in postprocessors.items():
                value = tuple(
                    alias_factory_subclass_from_arg(
                        PostProcessor, postprocessor)
                    for postprocessor in value)
                postprocessors[key] = value
            self.postprocessors = postprocessors
            postprocess_axis = kwargs.pop('postprocess_axis', -1)
            if not hasattr(postprocess_axis, '__len__'):
                postprocess_axis = (postprocess_axis,)
            if not hasattr(postprocess_axis, 'get'):
                post_dict = dict()
                for key in postprocessors:
                    post_dict[key] = postprocess_axis
                postprocess_axis = post_dict
            self.postprocess_axis = postprocess_axis
            super(_Wrapper, self).__init__(table, *additional_tables, **kwargs)

        def batch_generator(self, repeat=False):
            subsamples = self.num_sub != 1
            for batch in super(_Wrapper, self).batch_generator(repeat=repeat):
                if subsamples:
                    cur_batch = []
                    for sub_batch_idx, sub_batch in enumerate(batch):
                        for postprocessor, axis in zip(
                                self.postprocessors.get(
                                    sub_batch_idx, tuple()),
                                cycle(self.postprocess_axis.get(
                                    sub_batch_idx, tuple()))):
                            sub_batch = postprocessor.apply(
                                sub_batch, axis=axis, in_place=True)
                        cur_batch.append(sub_batch)
                    yield tuple(cur_batch)
                else:
                    for postprocessor, axis in zip(
                            self.postprocessors[0],
                            cycle(self.postprocess_axis[0])):
                        batch = postprocessor.apply(
                            batch, axis=axis, in_place=True)
                    yield batch
    _Wrapper.__doc__ = cls.__doc__ + post_process_wrapper.WRAPPED_DATA_DOC
    return _Wrapper


post_process_wrapper.WRAPPED_DATA_DOC = '''
This class has been wrapped using
``pydrobert.speech.post_process_wrapper``. A set of additional
parameters has been introduced:

Additional Parameters
---------------------
postprocessors : sequence or mapping, optional
    Some number of pydrobert.speech.pre.PostProcessor objects to apply to
    individual samples after batching. If a sequence is provided and
    num_sub == 1, the sequence of preprocessors will be applied to the
    batch in the order they are presented. If num_sub > 1, preprocessors
    will be applied to the first sub-batch per batch. If a mapping is
    provided, its values are sequences of preprocessors to apply to the
    sub-batch also indexed by the key
postprocess_axis : int or sequence or mapping
    What axis/axis to apply postprocessors to. If an int, every
    postprocessor will be applied on that axis. If a sequence,
    the sequences of postprocessors specified in `postprocessors` will
    be applied one-to-one on the sequence of axes. If there are fewer
    postprocessors than axes, the axes will repeat. A mapping can be
    used to specify different sequences for different sub-batches.
    Defaults to -1

Additional Attributes
---------------------
postprocessors : mapping
    The passed postprocessors, mapped to individual sub-batches
postprocess_axis : mapping
    The passed po
'''
