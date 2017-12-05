from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

from pydrobert.speech.corpus import post_process_wrapper
from pydrobert.speech.post import PostProcessor


class Sum(PostProcessor):
    '''Sum over the given axis

    Container for numpy.sum. Big whoop.
    '''
    aliases = {'sum'}

    def apply(self, features, axis=-1, in_place=False):
        return np.sum(features, axis=axis)

class Axis(PostProcessor):
    '''Return the passed axis'''

    aliases = {'axis'}

    def apply(self, features, axis=-1, in_place=False):
        return axis


def test_post_processor_wrapper(temp_file_1_name, temp_file_2_name):
    kaldi_io = pytest.importorskip('pydrobert.kaldi.io')
    kaldi_corpus = pytest.importorskip('pydrobert.kaldi.io.corpus')
    Wrapped = post_process_wrapper(kaldi_corpus.SequentialData)
    feats = []
    labels = []
    alphabet = tuple('abcdefgh')  # the expurgated version
    with kaldi_io.open(
            'ark:' + temp_file_1_name, 'dm', mode='w') as writer_1, \
            kaldi_io.open('ark:' + temp_file_2_name, 'tv', mode='w') \
            as writer_2:
        for sample_idx in range(100):
            feat_sample = np.random.random((np.random.randint(1, 100), 20))
            label_sample = tuple(np.random.choice(
                alphabet, size=np.random.randint(1, 10)).tolist())
            feats.append(feat_sample)
            labels.append(label_sample)
            writer_1.write('{:03d}'.format(sample_idx), feat_sample)
            writer_2.write('{:03d}'.format(sample_idx), label_sample)
    wrapped_1 = Wrapped(
        ('ark,s:' + temp_file_1_name, 'dm'),
        ('ark,s:' + temp_file_2_name, 'tv'),
        postprocessors=['sum'])
    n_samples = 0
    for ex_feat, ex_lab, (act_feat, act_lab) in zip(
            feats, labels, wrapped_1):
        assert ex_lab == act_lab
        assert act_feat.shape == ex_feat.shape[:-1]
        assert np.allclose(np.sum(ex_feat, axis=-1), act_feat)
        n_samples += 1
    assert n_samples == len(feats)
    wrapped_2 = Wrapped(
        ('ark,s:' + temp_file_1_name, 'dm'),
        ('ark,s:' + temp_file_2_name, 'tv'),
        postprocessors={0: ['sum'], 1: ['axis']},
        postprocess_axis=[0, -1],
    )
    n_samples = 0
    for ex_feat, (act_feat, act_lab) in zip(feats, wrapped_2):
        assert act_lab == 0
        assert ex_feat.shape[1:] == act_feat.shape
        assert np.allclose(np.sum(ex_feat, axis=0), act_feat)
        n_samples += 1
    assert n_samples == len(feats)
    wrapped_3 = Wrapped(
        ('ark,s:' + temp_file_1_name, 'dm'),
        ('ark,s:' + temp_file_2_name, 'tv'),
        postprocessors={0: ['sum', 'sum'], 1: ['axis', 'axis']},
        postprocess_axis={0: [1], 1: [2, -1]},
        batch_pad_mode='constant',
        batch_size=5,
    )
    n_samples = 0
    for act_feat, act_lab in wrapped_3:
        assert act_lab == -1
        ex_feats = feats[n_samples:n_samples + 5]
        # sum axis 1 then sum axis 1 means we've retained the batch axis
        assert np.allclose(
            np.stack([np.sum(s) for s in ex_feats]),
            act_feat
        )
        n_samples += len(ex_feats)
    assert n_samples == len(feats)
