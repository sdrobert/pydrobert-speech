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

'''Script-like functions intended to be accessed by command line'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import json
import logging
import sys

import numpy as np

import pydrobert.speech as speech

from pydrobert.speech.compute import FrameComputer
from pydrobert.speech.pre import PreProcessor
from pydrobert.speech.post import PostProcessor
from pydrobert.speech.util import alias_factory_subclass_from_arg
from pydrobert.speech.util import read_signal

from future.utils import raise_from

try:
    from pydrobert.kaldi.logging import kaldi_vlog_level_cmd_decorator
    from pydrobert.kaldi.logging import kaldi_logger_decorator
except ImportError:
    def kaldi_vlog_level_cmd_decorator(func):
        return func

    def kaldi_logger_decorator(func):
        return func

try:
    import torch.utils.data

    class _FeatureProcessorDataset(torch.utils.data.Dataset):
        # yields utt, feats

        def __init__(
                self, utt2path, preprocessors, computer, postprocessors,
                channel, force_as, seed):
            super(_FeatureProcessorDataset, self).__init__()
            self.utt_path = tuple(utt2path.items())
            self.preprocessors = preprocessors
            self.computer = computer
            self.postprocessors = postprocessors
            self.channel = channel
            self.force_as = force_as
            self.seed = seed

        def __len__(self):
            return len(self.utt_path)

        def __getitem__(self, idx):
            # torch.multiprocessing should copy numpy rng over workers, so
            # we need not worry about the rng changing underneath us
            np.random.seed(self.seed + idx)
            utt_id, path = self.utt_path[idx]
            try:
                signal = read_signal(
                    path, dtype=np.float64, force_as=self.force_as,
                    key=utt_id)
            except Exception as e:
                raise_from(IOError('Utterance {}:'.format(utt_id), e))
            if (
                    self.channel == -1 and
                    len(signal.shape) > 1 and
                    signal.shape[0] > 1):
                raise ValueError(
                    "Utterance {}: Channel is not specified but signal has "
                    "shape {}".format(utt_id, signal.shape))
            elif (
                    (self.channel != -1 and len(signal.shape) == 1) or
                    (self.channel >= signal.shape[0])):
                raise ValueError(
                    "Utterance {}: Channel specified as {} but signal has "
                    "shape {}".format(utt_id, self.channel, signal.shape))
            if len(signal.shape) != 1:
                signal = signal[self.channel]
            for preprocessor in self.preprocessors:
                signal = preprocessor.apply(signal, in_place=True)
            if self.computer is None:
                feats = signal[:, None]  # add "filter" dim to raw signal
            else:
                feats = self.computer.compute_full(signal)
            del signal
            for postprocessor in self.postprocessors:
                feats = postprocessor.apply(feats, in_place=True)
            feats = torch.FloatTensor(feats)
            return utt_id, feats
except ImportError:
    pass

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2019 Sean Robertson"
__all__ = [
    "compute_feats_from_kaldi_tables",
    "signals_to_torch_feat_dir",
]


def _json_type(string):
    '''Convert JSON string (or path to JSON file) to container hierarchy'''
    try:
        with open(string) as file_obj:
            return json.load(file_obj)
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Unable to parse file as json: "{}"'.format(string))
    except IOError:
        pass
    try:
        return json.loads(string)
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Unable to parse string as json: "{}"'.format(string))


def _nonneg_int_type(string):
    '''Convert to an int and make sure its nonnegative'''
    try:
        val = int(string)
        assert val >= 0
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            '{} is not a nonnegative integer'.format(string))
    return val


def _compute_feats_from_kaldi_tables_parse_args(args, logger):
    from pydrobert.kaldi.io.argparse import KaldiParser
    parser = KaldiParser(
        description=compute_feats_from_kaldi_tables.__doc__,
        add_verbose=True, logger=logger,
        version=speech.__version__,
    )
    parser.add_argument(
        'wav_rspecifier', type='kaldi_rspecifier',
        help='Input wave table rspecifier'
    )
    parser.add_argument(
        'feats_wspecifier', type='kaldi_wspecifier',
        help='Output feature table wspecifier'
    )
    parser.add_argument(
        'computer_config', type=_json_type,
        help='JSON file or string to configure a '
        '``pydrobert.speech.compute.FrameComputer`` object to calculate '
        'features with'
    )
    parser.add_argument(
        '--min-duration', type=float, default=0.0,
        help='Min duration of segments to process (in seconds)')
    parser.add_argument(
        '--channel', type=int, default=-1,
        help='Channel to draw audio from. Default is to assume mono'
    )
    parser.add_argument(
        '--preprocess', type=_json_type, default=tuple(),
        help='JSON list of configurations for '
        '``pydrobert.speech.pre.PreProcessor`` objects. Audio will be '
        'preprocessed in the same order as the list'
    )
    parser.add_argument(
        '--postprocess', type=_json_type, default=tuple(),
        help='JSON List of configurations for '
        '``pydrobert.speech.post.PostProcessor`` objects. Features will be '
        'postprocessed in the same order as the list'
    )
    parser.add_argument(
        '--seed', type=_nonneg_int_type, default=None,
        help='A random seed used for determinism. This affects operations '
        'like dithering. If unset, a seed will be generated at the moment'
    )
    return parser.parse_args(args)


@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def compute_feats_from_kaldi_tables(args=None):
    '''Store features from a kaldi archive in a kaldi archive

    This command is intended to replace Kaldi's [povey2011]_ series of
    ``compute-<something>-feats`` scripts in a Kaldi pipeline.
    '''
    from pydrobert.kaldi.logging import register_logger_for_kaldi
    from pydrobert.kaldi.io.enums import KaldiDataType
    from pydrobert.kaldi.io import open as kaldi_open
    logger = logging.getLogger(sys.argv[0])
    logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(logger)
    try:
        options = _compute_feats_from_kaldi_tables_parse_args(args, logger)
    except SystemExit as ex:
        return ex.code
    if options.seed is not None:
        np.random.seed(options.seed)
    # construct the computer
    try:
        computer = alias_factory_subclass_from_arg(
            FrameComputer, options.computer_config)
    except ValueError:
        logger.error('Failed to build computer:', exc_info=True)
        return 1
    # construct the preprocessors (if any)
    preprocessors = []
    try:
        if isinstance(options.preprocess, dict):
            preprocessors.append(alias_factory_subclass_from_arg(
                PreProcessor, options.preprocess))
        else:
            for element in options.preprocess:
                preprocessors.append(alias_factory_subclass_from_arg(
                    PreProcessor, element))
    except ValueError:
        logger.error('Failed to build preprocessor:', exc_info=True)
        return 1
    postprocessors = []
    try:
        if isinstance(options.postprocess, dict):
            postprocessors.append(alias_factory_subclass_from_arg(
                PostProcessor, options.postprocess))
        else:
            for element in options.postprocess:
                postprocessors.append(alias_factory_subclass_from_arg(
                    PostProcessor, element))
    except ValueError:
        logger.error('Failed to build postprocessor:', exc_info=True)
        return 1
    # open tables
    try:
        wav_reader = kaldi_open(
            options.wav_rspecifier, 'wm', value_style='bsd')
    except IOError:
        logger.error(
            'Could not read the wave table {}'.format(options.wav_rspecifier)
        )
        return 1
    try:
        feat_writer = kaldi_open(options.feats_wspecifier, 'bm', mode='w')
    except IOError:
        logger.error(
            'Could not open the feat table {} for writing'.format(
                options.feats_wspecifier))
        return 1
    num_utts, num_success = 0, 0
    for utt_id, (buff, samp_freq, duration) in wav_reader.items():
        num_utts += 1
        if duration < options.min_duration:
            logger.warn(
                'File: {} is too short ({:.2f} sec): producing no output'
                ''.format(utt_id, duration)
            )
            continue
        elif samp_freq != computer.bank.sampling_rate:
            logger.warn(
                'Sample frequency mismatch for file {}: you specified {:.2f} '
                'but data has {:.2f}: producing no output'
                ''.format(utt_id, computer.bank.sample_rate_hz, samp_freq)
            )
            continue
        cur_chan = options.channel
        if options.channel == -1 and buff.shape[0] > 1:
            logger.warning(
                'Channel is not specified but you have data with {} channels;'
                ' defaulting to zero'.format(buff.shape[0])
            )
            cur_chan = 0
        elif options.channel >= buff.shape[0]:
            logger.warn(
                'File with id {} has {} channels but you specified channel {},'
                ' producing no ouptut'.format(
                    utt_id, buff.shape[0], options.channel)
            )
            continue
        buff = buff[cur_chan].astype(np.float64, copy=False)
        for preprocessor in preprocessors:
            buff = preprocessor.apply(buff, in_place=True)
        feats = computer.compute_full(buff)
        if not KaldiDataType.BaseMatrix.is_double:
            feats = feats.astype(np.float32)
        feat_writer.write(utt_id, feats)
        if num_utts % 10 == 0:
            logger.info('Processed {} utterances'.format(num_utts))
        logger.log(9, 'Processed features for key {}'.format(utt_id))
        num_success += 1
    logger.info('Done {} out of {} utterances'.format(num_success, num_utts))
    feat_writer.close()
    wav_reader.close()
    return 0 if num_success else 1


def _signals_to_torch_feat_dir_parse_args(args):
    parser = argparse.ArgumentParser(
        description=signals_to_torch_feat_dir.__doc__)
    parser.add_argument(
        'map', type=argparse.FileType('r'),
        help='Path to the file containing ``utterance, path`` pairs')
    parser.add_argument(
        'computer_config', type=_json_type, nargs='?', default=None,
        help='JSON file or string to configure a '
        'pydrobert.speech.compute.FrameComputer object to calculate '
        'features with. If unspecified, the audio (with channels removed) '
        'will be stored directly with shape ``(S, 1)``, where ``S`` is the '
        'number of samples'
    )
    parser.add_argument(
        'dir', help='Directory to output features to. If the directory does '
        'not exist, it will be created'
    )
    parser.add_argument(
        '--channel', type=int, default=-1,
        help='Channel to draw audio from. Default is to assume mono'
    )
    parser.add_argument(
        '--preprocess', type=_json_type, default=tuple(),
        help='JSON list of configurations for '
        '``pydrobert.speech.pre.PreProcessor`` objects. Audio will be '
        'preprocessed in the same order as the list'
    )
    parser.add_argument(
        '--postprocess', type=_json_type, default=tuple(),
        help='JSON List of configurations for '
        '``pydrobert.speech.post.PostProcessor`` objects. Features will be '
        'postprocessed in the same order as the list'
    )
    parser.add_argument(
        '--force-as', default=None,
        choices={
            'tab', 'wav', 'hdf5', 'npy', 'npz', 'pt', 'sph', 'kaldi',
            'file',
        },
        help='Force the paths in `map` to be interpreted as a specific type '
        'of data. tab: kaldi table (key is utterance id); wav: wave file; '
        'hdf5: HDF5 archive (key is utterance id); npy: Numpy binary; npz: '
        'numpy archive (key is utterance id); pt: PyTorch binary; sph: NIST '
        'SPHERE file; kaldi: kaldi object; file: numpy.fromfile binary'
    )
    parser.add_argument(
        '--seed', type=_nonneg_int_type, default=None,
        help='A random seed used for determinism. This affects operations '
        'like dithering. If unset, a seed will be generated at the moment'
    )
    parser.add_argument(
        '--file-prefix', default='',
        help='The file prefix indicating a torch data file'
    )
    parser.add_argument(
        '--file-suffix', default='.pt',
        help='The file suffix indicating a torch data file'
    )
    parser.add_argument(
        '--num-workers', type=_nonneg_int_type, default=0,
        help='The number of workers simultaneously computing features. Should '
        'not affect determinism when used in tandem with "--seed". 0 means '
        'all work is done on the main thread'
    )
    return parser.parse_args(args)


def signals_to_torch_feat_dir(args=None):
    '''Convert a map of signals to a torch ``SpectDataSet``

    This command serves to process audio signals and convert them into a
    format that can be leveraged by ``SpectDataSet`` in
    `pydrobert-pytorch <https://github.com/sdrobert/pydrobert-pytorch>`_.
    It reads in a text file of format::

        <utt_id_1> <path_to_signal_1>
        <utt_id_2> <path_to_signal_2>
        ...

    computes features according to passed-in settings, and stores them in the
    target directory as::

        dir/
            <file_prefix><utt_id_1><file_suffix>
            <file_prefix><utt_id_2><file_suffix>
            ...

    Each signal is read using the utility
    ``pydrobert.speech.util.read_signal()``, which is a bit slow, but very
    robust to different file types (such as wave files, hdf5, numpy binaries,
    or Pytorch binaries). A signal is expected to have shape ``(C, S)``, where
    ``C`` is some number of channels and ``S`` is some number of samples. The
    signal can have shape ``(S,)`` if the flag ``--channels=-1``.

    Features are output as ``torch.FloatTensor`` of shape ``(T, F)``, where
    ``T`` is some number of frames and ``F`` is some number of filters.

    Warning
    -------
    No checks are performed to ensure that read signals match the feature
    computer's sampling rate (this info may not even exist for some
    sources).
    '''
    try:
        options = _signals_to_torch_feat_dir_parse_args(args)
    except SystemExit as ex:
        return ex.code
    try:
        import torch
        import torch.utils.data
    except ImportError:
        print(
            'signals-to-torch-feat-dir requires a PyTorch installation',
            file=sys.stderr)
        return 1
    if options.seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)
    else:
        seed = options.seed
    utt2path = dict()
    for line_no, line in enumerate(options.map):
        line = line.strip()
        if not line:
            continue
        ls = line.split(' ')
        if len(ls) < 2:
            print(
                'Line {} of {}: not of format <utt_id> <path>'.format(
                    line_no + 1, options.map.name),
                file=sys.stderr)
            return 1
        utt_id = ls[0]
        if utt_id in utt2path:
            print(
                'Line {} of {}: "{}" already exists as utterance'.format(
                    line_no + 1, options.map.name, utt_id),
                file=sys.stderr)
            return 1
        utt2path[utt_id] = ' '.join(ls[1:])
    if options.computer_config is None:
        computer = None
    else:
        computer = alias_factory_subclass_from_arg(
            FrameComputer, options.computer_config)
    preprocessors = []
    if isinstance(options.preprocess, dict):
        preprocessors.append(alias_factory_subclass_from_arg(
            PreProcessor, options.preprocess))
    else:
        for element in options.preprocess:
            preprocessors.append(alias_factory_subclass_from_arg(
                PreProcessor, element))
    postprocessors = []
    if isinstance(options.postprocess, dict):
        postprocessors.append(alias_factory_subclass_from_arg(
            PostProcessor, options.postprocess))
    else:
        for element in options.postprocess:
            postprocessors.append(alias_factory_subclass_from_arg(
                PostProcessor, element))
    # we use a dataset to take advantage of torch's multiprocessing
    dataset = _FeatureProcessorDataset(
        utt2path, preprocessors, computer, postprocessors, options.channel,
        options.force_as, seed)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=options.num_workers)
    if not os.path.isdir(options.dir):
        os.makedirs(options.dir)
    for utt_ids, feats in loader:
        utt_id, feat = utt_ids[0], feats[0]
        torch.save(feat, os.path.join(
            options.dir, options.file_prefix + utt_id + options.file_suffix))
    return 0
