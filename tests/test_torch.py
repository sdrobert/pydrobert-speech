import os
import json

import pytest
import numpy as np

try:
    import torch

    skip = False
except ImportError:
    skip = True

import pydrobert.speech.compute as compute

from pydrobert.speech.torch import PyTorchSTFTFrameComputer
from pydrobert.speech.alias import alias_factory_subclass_from_arg

pytestmark = pytest.mark.skipif(skip, reason="Could not import pytorch")


def test_stft_frame_computer_wrapper():
    torch.manual_seed(1)
    signal = torch.randn(16_000)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    with open(os.path.join(data_dir, "fbank.json")) as json_file:
        computer_numpy = alias_factory_subclass_from_arg(
            compute.FrameComputer, json.load(json_file)
        )
    computer_pytorch = PyTorchSTFTFrameComputer.from_numpy_frame_computer(
        computer_numpy
    )
    exp = computer_numpy.compute_full(signal.numpy())
    with torch.no_grad():
        act = computer_pytorch(signal).numpy()
    assert np.allclose(exp, act)
