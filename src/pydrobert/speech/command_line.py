# Copyright 2023 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script-like functions intended to be accessed by command line"""


import os
import argparse
import logging
import sys
from typing import Optional, Sequence

import numpy as np

from .. import speech
from . import config

from .compute import FrameComputer, STFTFrameComputer, SIFrameComputer
from .pre import Dither, PreProcessor, Preemphasize
from .post import PostProcessor
from .alias import alias_factory_subclass_from_arg
from .util import read_signal

try:
    from pydrobert.kaldi.logging import kaldi_vlog_level_cmd_decorator  # type: ignore
    from pydrobert.kaldi.logging import kaldi_logger_decorator  # type: ignore
except ImportError:

    def kaldi_vlog_level_cmd_decorator(func):
        return func

    def kaldi_logger_decorator(func):
        return func


try:
    from ruamel.yaml import YAML

    def _load_config(string: str):
        return YAML(typ="safe").load(string)

    _HAVE_YAML = True

except ImportError:
    from json import loads

    _load_config = loads
    _HAVE_YAML = False

_EPILOGUE = """New in version 0.4.0: if ruamel.yaml is installed
(https://yaml.readthedocs.io/en/latest/), JSON arguments will be parsed as YAML 1.2
by default. As JSON is valid YAML 1.2, you can continue to use JSON for configurations.
"""

try:
    import torch.utils.data

    from .torch import (
        PyTorchSTFTFrameComputer,
        PyTorchSIFrameComputer,
        PyTorchDither,
        PyTorchPreemphasize,
        PyTorchPostProcessorWrapper,
    )

    class _FeatureProcessorDataset(torch.utils.data.Dataset):
        # yields utt, feats

        def __init__(
            self,
            utt2path,
            preprocessors,
            computer,
            postprocessors,
            channel,
            force_as,
            seed,
        ):
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

        @torch.no_grad()
        def __getitem__(self, idx):
            torch.manual_seed(self.seed + idx)
            utt_id, path = self.utt_path[idx]
            try:
                signal = read_signal(
                    path, dtype=np.float64, force_as=self.force_as, key=utt_id
                )
            except Exception as e:
                raise IOError(f"Utterance {utt_id}: {e}") from e
            if self.channel == -1 and signal.ndim > 1 and signal.shape[0] > 1:
                raise ValueError(
                    "Utterance {}: Channel is not specified but signal has "
                    "shape {}".format(utt_id, signal.shape)
                )
            elif (self.channel != -1 and signal.ndim == 1) or (
                self.channel >= signal.shape[0]
            ):
                raise ValueError(
                    "Utterance {}: Channel specified as {} but signal has "
                    "shape {}".format(utt_id, self.channel, signal.shape)
                )
            if signal.ndim != 1:
                signal = signal[self.channel]
            signal = torch.from_numpy(signal)
            for preprocessor in self.preprocessors:
                signal = preprocessor(signal)
            if self.computer is None:
                feats = signal.unsqueeze(1)
            else:
                feats = self.computer(signal)
            del signal
            for postprocessor in self.postprocessors:
                feats = postprocessor(feats)
            return utt_id, feats.float()

except ImportError:
    pass

__all__ = [
    "compute_feats_from_kaldi_tables",
    "signals_to_torch_feat_dir",
]


def _config_type(string: str):
    """Convert JSON string (or path to file) to container hierarchy"""
    name = string
    try:
        with open(string) as file_obj:
            string = file_obj.read()
    except IOError:
        pass
    try:
        return _load_config(string)
    except Exception as e:
        if _HAVE_YAML:
            msg = f"Unable to parse '{name}' as JSON or YAML"
        else:
            msg = f"Unable to parse '{name}' as JSON"
            if name.endswith(".yaml"):
                msg += f". This could be a YAML file. Install ruamel.yaml to try it"
        raise ValueError(msg) from e


def _nonneg_int_type(string):
    """Convert to an int and make sure its nonnegative"""
    try:
        val = int(string)
        assert val >= 0
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError(
            "{} is not a nonnegative integer".format(string)
        )
    return val


def _compute_feats_from_kaldi_tables_parse_args(args, logger):
    from pydrobert.kaldi.io.argparse import KaldiParser  # type: ignore

    parser = KaldiParser(
        description=compute_feats_from_kaldi_tables.__doc__,
        add_verbose=True,
        logger=logger,
        version=speech.__version__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOGUE,
    )
    parser.add_argument(
        "wav_rspecifier", type="kaldi_rspecifier", help="Input wave table rspecifier"
    )
    parser.add_argument(
        "feats_wspecifier",
        type="kaldi_wspecifier",
        help="Output feature table wspecifier",
    )
    parser.add_argument(
        "computer_config",
        type=_config_type,
        help="JSON file or string to configure a "
        "'pydrobert.speech.compute.FrameComputer' object to calculate "
        "features with",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Min duration of segments to process (in seconds)",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=-1,
        help="Channel to draw audio from. Default is to assume mono",
    )
    parser.add_argument(
        "--preprocess",
        type=_config_type,
        default=tuple(),
        help="JSON list of configurations for "
        "'pydrobert.speech.pre.PreProcessor' objects. Audio will be "
        "preprocessed in the same order as the list",
    )
    parser.add_argument(
        "--postprocess",
        type=_config_type,
        default=tuple(),
        help="JSON list of configurations for "
        "'pydrobert.speech.post.PostProcessor' objects. Features will be "
        "postprocessed in the same order as the list",
    )
    parser.add_argument(
        "--seed",
        type=_nonneg_int_type,
        default=None,
        help="A random seed used for determinism. This affects operations "
        "like dithering. If unset, a seed will be generated at the moment",
    )
    return parser.parse_args(args)


@kaldi_vlog_level_cmd_decorator
@kaldi_logger_decorator
def compute_feats_from_kaldi_tables(args: Optional[Sequence[str]] = None) -> None:
    """Store features from a kaldi archive in a kaldi archive

    This command is intended to replace Kaldi's (https://kaldi-asr.org/) series of
    "compute-<something>-feats" scripts in a Kaldi pipeline.
    """
    from pydrobert.kaldi.logging import register_logger_for_kaldi  # type: ignore
    from pydrobert.kaldi.io.enums import KaldiDataType  # type: ignore
    from pydrobert.kaldi.io import open as kaldi_open  # type: ignore

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
            FrameComputer, options.computer_config
        )
    except ValueError:
        logger.error("Failed to build computer:", exc_info=True)
        return 1
    # construct the preprocessors (if any)
    preprocessors = []
    try:
        if isinstance(options.preprocess, dict):
            preprocessors.append(
                alias_factory_subclass_from_arg(PreProcessor, options.preprocess)
            )
        else:
            for element in options.preprocess:
                preprocessors.append(
                    alias_factory_subclass_from_arg(PreProcessor, element)
                )
    except ValueError:
        logger.error("Failed to build preprocessor:", exc_info=True)
        return 1
    postprocessors = []
    try:
        if isinstance(options.postprocess, dict):
            postprocessors.append(
                alias_factory_subclass_from_arg(PostProcessor, options.postprocess)
            )
        else:
            for element in options.postprocess:
                postprocessors.append(
                    alias_factory_subclass_from_arg(PostProcessor, element)
                )
    except ValueError:
        logger.error("Failed to build postprocessor:", exc_info=True)
        return 1
    # open tables
    try:
        wav_reader = kaldi_open(options.wav_rspecifier, "wm", value_style="bsd")
    except IOError:
        logger.error("Could not read the wave table {}".format(options.wav_rspecifier))
        return 1
    try:
        feat_writer = kaldi_open(options.feats_wspecifier, "bm", mode="w")
    except IOError:
        logger.error(
            "Could not open the feat table {} for writing".format(
                options.feats_wspecifier
            )
        )
        return 1
    num_utts, num_success = 0, 0
    for utt_id, (buff, samp_freq, duration) in list(wav_reader.items()):
        num_utts += 1
        if duration < options.min_duration:
            logger.warn(
                "File: {} is too short ({:.2f} sec): producing no output"
                "".format(utt_id, duration)
            )
            continue
        elif samp_freq != computer.bank.sampling_rate:
            logger.warn(
                "Sample frequency mismatch for file {}: you specified {:.2f} "
                "but data has {:.2f}: producing no output"
                "".format(utt_id, computer.bank.sample_rate_hz, samp_freq)
            )
            continue
        cur_chan = options.channel
        if options.channel == -1 and buff.shape[0] > 1:
            logger.warning(
                "Channel is not specified but you have data with {} channels;"
                " defaulting to zero".format(buff.shape[0])
            )
            cur_chan = 0
        elif options.channel >= buff.shape[0]:
            logger.warn(
                "File with id {} has {} channels but you specified channel {},"
                " producing no ouptut".format(utt_id, buff.shape[0], options.channel)
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
            logger.info("Processed {} utterances".format(num_utts))
        logger.log(9, "Processed features for key {}".format(utt_id))
        num_success += 1
    logger.info("Done {} out of {} utterances".format(num_success, num_utts))
    feat_writer.close()
    wav_reader.close()
    return 0 if num_success else 1


def _signals_to_torch_feat_dir_parse_args(args):
    parser = argparse.ArgumentParser(
        description=signals_to_torch_feat_dir.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOGUE,
    )
    parser.add_argument(
        "map",
        type=argparse.FileType("r"),
        help="Path to the file containing (<utterance>, <path>) pairs",
    )
    parser.add_argument(
        "computer_config",
        type=_config_type,
        nargs="?",
        default=None,
        help="JSON file or string to configure a "
        "pydrobert.speech.compute.FrameComputer object to calculate features with. If "
        "unspecified, the audio (with channels removed)  will be stored directly with "
        "shape (S, 1), where S is the number of samples",
    )
    parser.add_argument(
        "dir",
        help="Directory to output features to. If the directory does not exist, it "
        "will be created",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=-1,
        help="Channel to draw audio from. Default is to assume mono",
    )
    parser.add_argument(
        "--preprocess",
        type=_config_type,
        default=tuple(),
        help="JSON list of configurations for "
        "'pydrobert.speech.pre.PreProcessor' objects. Audio will be preprocessed in "
        "the same order as the list",
    )
    parser.add_argument(
        "--postprocess",
        type=_config_type,
        default=tuple(),
        help="JSON list of configurations for "
        "'pydrobert.speech.post.PostProcessor' objects. Features will be postprocessed "
        "in the same order as the list",
    )
    parser.add_argument(
        "--force-as",
        default=None,
        choices={
            "table",
            "wav",
            "hdf5",
            "npy",
            "npz",
            "pt",
            "sph",
            "kaldi",
            "file",
            "soundfile",
        }
        | config.SOUNDFILE_SUPPORTED_FILE_TYPES,
        help="Force the paths in 'map' to be interpreted as a specific type "
        "of data. table: kaldi table (key is utterance id); wav: wave file; "
        "hdf5: HDF5 archive (key is utterance id); npy: Numpy binary; npz: "
        "numpy archive (key is utterance id); pt: PyTorch binary; sph: NIST "
        "SPHERE file; kaldi: kaldi object; file: numpy.fromfile binary. soundfile: "
        "force soundfile processing.",
    )
    parser.add_argument(
        "--seed",
        type=_nonneg_int_type,
        default=None,
        help="A random seed used for determinism. This affects operations "
        "like dithering. If unset, a seed will be generated at the moment",
    )
    parser.add_argument(
        "--file-prefix", default="", help="The file prefix indicating a torch data file"
    )
    parser.add_argument(
        "--file-suffix",
        default=".pt",
        help="The file suffix indicating a torch data file",
    )
    parser.add_argument(
        "--num-workers",
        type=_nonneg_int_type,
        default=0,
        help="The number of workers simultaneously computing features. Should not "
        "affect determinism when used in tandem with --seed. '0' means all work is "
        "done on the main thread",
    )
    parser.add_argument(
        "--manifest",
        type=argparse.FileType("a+"),
        default=None,
        help="If specified, a list of utterances which have already been computed "
        "will be stored in this file. Utterances already listed in the file will be "
        "not be computed. Useful for resuming computations after an unexpected "
        "termination",
    )
    return parser.parse_args(args)


def signals_to_torch_feat_dir(args=None):
    """Convert a map of signals to a torch SpectDataSet

    This command serves to process audio signals and convert them into a format that can
    be leveraged by "SpectDataSet" in "pydrobert-pytorch"
    (https://github.com/sdrobert/pydrobert-pytorch). It reads in a text file of
    format

        <utt_id_1> <path_to_signal_1>
        <utt_id_2> <path_to_signal_2>
        ...

    computes features according to passed-in settings, and stores them in the
    target directory as

        dir/
            <file_prefix><utt_id_1><file_suffix>
            <file_prefix><utt_id_2><file_suffix>
            ...

    Each signal is read using the utility "pydrobert.speech.util.read_signal()", which
    is a bit slow, but very robust to different file types (such as wave files, hdf5,
    numpy binaries, or Pytorch binaries). A signal is expected to have shape (C, S),
    where C is some number of channels and S is some number of samples. The
    signal can have shape (S,) if the flag "--channels" is set to "-1".

    Features are output as "torch.FloatTensor" of shape "(T, F)", where "T" is some
    number of frames and "F" is some number of filters.

    No checks are performed to ensure that read signals match the feature computer's
    sampling rate (this info may not even exist for some sources).
    """
    try:
        options = _signals_to_torch_feat_dir_parse_args(args)
    except SystemExit as ex:
        return ex.code
    try:
        import torch
        import torch.utils.data
    except ImportError:
        print(
            "signals-to-torch-feat-dir requires a PyTorch installation", file=sys.stderr
        )
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
        ls = line.split(" ")
        if len(ls) < 2:
            print(
                "Line {} of {}: not of format <utt_id> <path>".format(
                    line_no + 1, options.map.name
                ),
                file=sys.stderr,
            )
            return 1
        utt_id = ls[0]
        if utt_id in utt2path:
            print(
                'Line {} of {}: "{}" already exists as utterance'.format(
                    line_no + 1, options.map.name, utt_id
                ),
                file=sys.stderr,
            )
            return 1
        utt2path[utt_id] = " ".join(ls[1:])
    if options.manifest is not None:
        options.manifest.seek(0)
        for line in options.manifest:
            utt2path.pop(line.strip(), None)
    if options.computer_config is None:
        computer = None
    else:
        computer = alias_factory_subclass_from_arg(
            FrameComputer, options.computer_config
        )
        if isinstance(computer, STFTFrameComputer):
            computer = PyTorchSTFTFrameComputer.from_stft_frame_computer(computer)
        elif isinstance(computer, SIFrameComputer):
            computer = PyTorchSIFrameComputer.from_si_frame_computer(computer)
        else:
            raise NotImplementedError
    preprocessors = []
    if isinstance(options.preprocess, dict):
        preprocessors.append(
            alias_factory_subclass_from_arg(PreProcessor, options.preprocess)
        )
    else:
        for element in options.preprocess:
            preprocessors.append(alias_factory_subclass_from_arg(PreProcessor, element))
    for i, preprocessor in enumerate(preprocessors):
        if isinstance(preprocessor, Dither):
            preprocessors[i] = PyTorchDither.from_dither(preprocessor)
        elif isinstance(preprocessor, Preemphasize):
            preprocessors[i] = PyTorchPreemphasize.from_preemphasize(preprocessor)
        else:
            raise NotImplementedError
    postprocessors = []
    if isinstance(options.postprocess, dict):
        postprocessors.append(
            alias_factory_subclass_from_arg(PostProcessor, options.postprocess)
        )
    else:
        for element in options.postprocess:
            postprocessors.append(
                alias_factory_subclass_from_arg(PostProcessor, element)
            )
    postprocessors = [
        PyTorchPostProcessorWrapper.from_postprocessor(p) for p in postprocessors
    ]
    # we use a dataset to take advantage of torch's multiprocessing
    dataset = _FeatureProcessorDataset(
        utt2path,
        preprocessors,
        computer,
        postprocessors,
        options.channel,
        options.force_as,
        seed,
    )
    loader = torch.utils.data.DataLoader(dataset, num_workers=options.num_workers)
    if not os.path.isdir(options.dir):
        os.makedirs(options.dir)
    for utt_ids, feats in loader:
        utt_id, feat = utt_ids[0], feats[0]
        torch.save(
            feat,
            os.path.join(
                options.dir, options.file_prefix + utt_id + options.file_suffix
            ),
        )
        if options.manifest is not None:
            print(utt_id, file=options.manifest)
    return 0
