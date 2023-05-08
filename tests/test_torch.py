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


@pytest.mark.parametrize("include_energy", [True, False], ids=["energy", "noenergy"])
@pytest.mark.parametrize("jit_type", ["script", "trace", "none"])
def test_pytorch_stft_frame_computer(include_energy, jit_type):
    torch.manual_seed(1)
    signal = torch.randn(16_000)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    with open(os.path.join(data_dir, "fbank.json")) as json_file:
        json_ = json.load(json_file)
    json_["include_energy"] = include_energy
    computer_numpy = alias_factory_subclass_from_arg(compute.FrameComputer, json_)
    computer_pytorch = PyTorchSTFTFrameComputer.from_stft_frame_computer(computer_numpy)
    if jit_type == "script":
        computer_pytorch = torch.jit.script(computer_pytorch)
    elif jit_type == "trace":
        computer_pytorch = torch.jit.trace(computer_pytorch, (torch.empty(1),))
    exp = computer_numpy.compute_full(signal.numpy())
    with torch.no_grad():
        act = computer_pytorch(signal).numpy()
    assert np.allclose(exp, act, atol=1e-5)
