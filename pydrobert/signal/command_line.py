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
from pydrobert.signal.util import alias_factory_subclass_from_arg

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

def kaldi_rspecifier_type(string):
    '''Make sure string is a valid rspecifier

    Raises
    ------
    ImportError
        If pydrobert-kaldi is not installed
    '''
    from pydrobert.kaldi.io.enums import TableType
    from pydrobert.kaldi.io.util import parse_kaldi_input_path
    table_type, _, _, _ = parse_kaldi_input_path(string)
    if table_type == TableType.NotATable:
        raise argparse.ArgumentTypeError(
            '{} is not a valid rspecifier'.format(string))
    return string

def kaldi_wspecifier_type(string):
    '''Make sure string is a valid wspecifier

    Raises
    ------
    ImportError
        If pydrobert-kaldi is not installed
    '''
    from pydrobert.kaldi.io.enums import TableType
    from pydrobert.kaldi.io.util import parse_kaldi_output_path
    table_type, _, _, _ = parse_kaldi_output_path(string)
    if table_type == TableType.NotATable:
        raise argparse.ArgumentTypeError(
            '{} is not a valid wspecifier'.format(string))
    return string

def kaldi_vlog_level_cmd_decorator(func):
    '''Decorator to rename, then revert, level names according to Kaldi [1]_

    See pydrobert.kaldi for the conversion chart. After the return of
    the function, the level names before the call are reverted. This
    function is insensitive to renaming while the function executes

    .. [1] Povey, D., et al (2011). The Kaldi Speech Recognition
           Toolkit. ASRU
    '''
    def _new_cmd(*args, **kwargs):
        __doc__ = func.__doc__
        old_level_names = [logging.getLevelName(0)]
        for level in range(1, 10):
            old_level_names.append(logging.getLevelName(level))
            logging.addLevelName(level, 'VLOG [{:d}]'.format(11 - level))
        for level in range(10, 51):
            old_level_names.append(logging.getLevelName(level))
            if level // 10 == 1:
                logging.addLevelName(level, 'VLOG [1]')
            elif level // 10 == 2:
                logging.addLevelName(level, 'LOG')
            elif level // 10 == 3:
                logging.addLevelName(level, 'WARNING')
            elif level // 10 == 4:
                logging.addLevelName(level, 'ERROR')
            elif level // 10 == 5:
                logging.addLevelName(level, 'ASSERTION_FAILED ')
        try:
            ret = func(*args, **kwargs)
        finally:
            for level, name in enumerate(old_level_names):
                logging.addLevelName(level, name)
        return ret
    return _new_cmd

def _kaldi_argparse_boilerplate(descr, parent, args):
    '''Boilerplate surrounding argument creation for kaldi-like scripts

    Arguments
    ---------
    descr : str
        Passed to the parser
    parent : argparse.ArgumentParser
        The parent parser from which all arguments are taken. Parent
        should not have help flag
    args
        The arguments to parse

    Returns
    ------
    tuple or None
        If there was an error, returns None. Otherwise, a pair of
        (namespace, logger)
    '''
    prog_name = path.basename(sys.argv[0])
    class _StdErrHelpAction(argparse.Action):
        def __init__(self, *args, **kwargs):
            self.logger = None
            super(_StdErrHelpAction, self).__init__(*args, **kwargs)
        def __call__(self, parser, *args, **kwargs):
            parser.print_help(file=sys.stderr)
            parser.exit()
    logging.captureWarnings(True)
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(logging.Formatter(
        '%(levelname)s (' + prog_name + '[0.0]:%(funcName)s():'
        '%(filename)s:%(lineno)d) %(message)s'
    ))
    i_logger = logging.getLogger('{}.{}.init'.format(__name__, prog_name))
    i_logger.addHandler(stderr_handler)
    parser = argparse.ArgumentParser(
        description=descr,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False, parents=[parent],
        fromfile_prefix_chars='@', # for kaldi-like config files
    )
    parser.add_argument(
        '-h', '--help', action=_StdErrHelpAction, default=argparse.SUPPRESS,
        nargs=0,
    )
    parser.add_argument(
        '-v', '--verbose', type=int, choices=[x for x in range(-3, 10)],
        default=0, help='Verbose level (higher->more logging). ',
    )
    try:
        namespace = parser.parse_args(args)
    except SystemExit:
        i_logger.error("Failed to parse arguments")
        return None
    try:
        from pydrobert.kaldi.logging import KaldiLogger
    except ImportError:
        i_logger.error('Unable to import pydrobert.kaldi.')
        return None
    old_logger_class = logging.getLoggerClass()
    logging.setLoggerClass(KaldiLogger)
    k_logger = logging.getLogger('{}.{}'.format(__name__, prog_name))
    logging.setLoggerClass(old_logger_class)
    k_logger.addHandler(stderr_handler)
    i_logger.removeHandler(stderr_handler)
    if namespace.verbose <= 1:
        k_logger.setLevel(namespace.verbose * -10 + 20)
    else:
        k_logger.setLevel(11 - namespace.verbose)
    return namespace, k_logger

@kaldi_vlog_level_cmd_decorator
def compute_feats_from_kaldi_tables(args=None):
    '''Store features from a kaldi archive in a kaldi archive

    This command is intended to replace Kaldi's [1]_ series of
    ``compute-<something>-feats`` scripts in a Kaldi pipeline.

    .. [1] Povey, D., et al (2011). The Kaldi Speech Recognition
           Toolkit. ASRU
    '''
    # parse arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        'wav_rspecifier', type=kaldi_rspecifier_type,
        help='Input wave table rspecifier'
    )
    parser.add_argument(
        'feats_wspecifier', type=kaldi_wspecifier_type,
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
    ret = _kaldi_argparse_boilerplate(
        compute_feats_from_kaldi_tables.__doc__,
        parser,
        args,
    )
    if ret is None:
        return 0
    else:
        namespace, logger = ret
    # construct the computer
    try:
        computer = alias_factory_subclass_from_arg(
            FrameComputer, namespace.computer_config)
    except ValueError:
        logger.error('Failed to build computer:', exc_info=True)
        return 1
    from pydrobert.kaldi.io import open as io_open
    # open tables
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
        buff = buff[cur_chan]
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

