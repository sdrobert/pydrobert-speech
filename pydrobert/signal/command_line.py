'''Script-like functions intended to be accessed by command line'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import sys

from os import path

from pydrobert.signal.compute import FrameComputer
from pydrobert.signal.pre import PreProcessor
from pydrobert.signal.util import alias_factory_subclass_from_arg

try:
    from pydrobert.kaldi.command_line import kaldi_vlog_level_cmd_decorator
    from pydrobert.kaldi.command_line import kaldi_logger_decorator
except ImportError:
    def kaldi_vlog_level_cmd_decorator(func):
        return func
    def kaldi_logger_decorator(func):
        return func

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2017 Sean Robertson"

def json_type(string):
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

def nonneg_int_type(string):
    '''Convert to an int and make sure its nonnegative'''
    try:
        val = int(string)
        assert val >= 0
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            '{} is not a nonnegative integer'.format(string))
    return val

@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def compute_feats_from_kaldi_tables(args=None):
    '''Store features from a kaldi archive in a kaldi archive

    This command is intended to replace Kaldi's [1]_ series of
    ``compute-<something>-feats`` scripts in a Kaldi pipeline.

    .. [1] Povey, D., et al (2011). The Kaldi Speech Recognition
           Toolkit. ASRU
    '''
    from pydrobert.kaldi.command_line import KaldiParser
    from pydrobert.kaldi.logging import register_logger_for_kaldi
    logger = logging.getLogger(sys.argv[0])
    logger.addHandler(logging.StreamHandler())
    register_logger_for_kaldi(logger)
    # parse arguments
    parser = KaldiParser(
        description=compute_feats_from_kaldi_tables.__doc__,
        add_verbose=True, logger=logger,
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
        'computer_config', type=json_type,
        help='JSON file or string to configure a '
        'pydrobert.signal.compute.FrameComputer object to calculate '
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
        '--preprocess', type=json_type, default=tuple(),
        help='JSON list of configurations for pydrobert.signal.pre.PreProcessor'
        ' objects. Audio will be preprocessed in the same order as the list'
    )
    try:
        options = parser.parse_args(args)
    except SystemExit as ex:
        return ex.code
    # construct the computer
    try:
        computer = alias_factory_subclass_from_arg(
            FrameComputer, namespace.computer_config)
    except ValueError:
        logger.error('Failed to build computer:', exc_info=True)
        return 1
    # construct the preprocessors (if any)
    preprocessors = []
    try:
        if isinstance(namespace.preprocess, dict):
            preprocessors.append(alias_factory_subclass_from_arg(
                PreProcessor, namespace.preprocess))
        else:
            for element in namespace.preprocess:
                preprocessors.append(alias_factory_subclass_from_arg(
                    PreProcessor, element))
    except ValueError:
        logger.error('Failed to build preprocessor:', exc_info=True)
        return 1
    # open tables
    from pydrobert.kaldi.io import open as io_open
    try:
        wav_reader = io_open(
            namespace.wav_rspecifier, 'wm', value_style='bsd')
    except IOError:
        logger.error(
            'Could not read the wave table {}'.format(namespace.wav_rspecifier)
        )
        return 1
    try:
        feat_writer = io_open(namespace.feats_wspecifier, 'bm', mode='w')
    except IOError:
        logger.error(
            'Could not open the feat table {} for writing'.format(
                namespace.feats_wspecifier))
        return 1
    num_utts, num_success = 0, 0
    for utt_id, (buff, samp_freq, duration) in wav_reader.items():
        num_utts += 1
        if duration < namespace.min_duration:
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
        cur_chan = namespace.channel
        if namespace.channel == -1 and buff.shape[0] > 1:
            logger.warning(
                'Channel is not specified but you have data with {} channels;'
                ' defaulting to zero'.format(buff.shape[0])
            )
            cur_chan = 0
        elif namespace.channel >= buff.shape[0]:
            logger.warn(
                'File with id {} has {} channels but you specified channel {},'
                ' producing no ouptut'.format(
                    utt_id, buff.shape[0], namespace.channel)
            )
            continue
        buff = buff[cur_chan].astype(np.float64, copy=False)
        for preprocessor in preprocessors:
            buff = preprocessor.apply(buff, in_place=True)
        feats = computer.compute_full(buff)
        feat_writer.write(utt_id, feats)
        if num_utts % 10 == 0:
            logger.info('Processed {} utterances'.format(num_utts))
        logger.log(9, 'Processed features for key {}'.format(utt_id))
        num_success += 1
    logger.info('Done {} out of {} utterances'.format(num_success, num_utts))
    feat_writer.close()
    wav_reader.close()
    return 0 if num_success else 1

