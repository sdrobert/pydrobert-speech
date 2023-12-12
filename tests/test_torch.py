import os
import json

import pytest
import numpy as np

try:
    import torch

    from pydrobert.speech.torch import *

    skip = False
except ImportError:
    skip = True

import pydrobert.speech.compute as compute

from pydrobert.speech.pre import *
from pydrobert.speech.post import *
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


@pytest.mark.parametrize("jit_type", ["script", "trace", "none"])
def test_pytorch_dither(jit_type):
    torch.manual_seed(2)
    std = 2.0
    dither = PyTorchDither.from_dither(Dither(std))
    if jit_type == "script":
        dither = torch.jit.script(dither)
    elif jit_type == "trace":
        dither = torch.jit.trace(dither, (torch.empty(1),), check_trace=False)
    signal = dither(torch.zeros(1_000_000))
    assert torch.isclose(signal.mean(), torch.tensor(0.0), atol=1e-2)
    assert torch.isclose(signal.std(), torch.tensor(std), atol=1e-2)


@pytest.mark.parametrize("jit_type", ["script", "trace", "none"])
def test_pytorch_preemphasize(jit_type):
    torch.manual_seed(3)
    coeff = 3.0
    signal = torch.randn(16_000)
    preemphasize_numpy = Preemphasize(coeff)
    preemphasize_pytorch = PyTorchPreemphasize.from_preemphasize(preemphasize_numpy)
    if jit_type == "script":
        preemphasize_pytorch = torch.jit.script(preemphasize_pytorch)
    elif jit_type == "trace":
        preemphasize_pytorch = torch.jit.trace(preemphasize_pytorch, (torch.empty(1),))
    exp = preemphasize_numpy.apply(signal.numpy())
    act = preemphasize_pytorch(signal).numpy()
    assert np.allclose(exp, act, atol=1e-5)


@pytest.mark.parametrize(
    "postprocessor",
    [Standardize(), Deltas(1), Stack(3)],
    ids=["standardize", "deltas", "stack"],
)
@pytest.mark.parametrize(
    "jit_type",
    [
        pytest.param("script", marks=pytest.mark.xfail(reason="No pytorch impl")),
        pytest.param("trace", marks=pytest.mark.xfail(reason="No pytorch impl")),
        "none",
    ],
)
def test_pytorch_post_processor_wrapper(postprocessor, jit_type):
    torch.manual_seed(4)
    spect = torch.randn(300, 10)
    wrapper = PyTorchPostProcessorWrapper.from_postprocessor(postprocessor)
    exp = torch.tensor(postprocessor.apply(spect.numpy()))
    if jit_type == "script":
        wrapper = torch.jit.script(wrapper)
    elif jit_type == "trace":
        with pytest.warns(torch.jit.TracerWarning):
            wrapper = torch.jit.trace(
                wrapper, (torch.empty(1, 1),), strict=False, check_trace=False
            )
    act = wrapper(spect)
    assert exp.shape == act.shape
    assert torch.allclose(exp.float(), act)


@pytest.mark.parametrize("include_energy", [True, False], ids=["energy", "noenergy"])
@pytest.mark.parametrize(
    "jit_type",
    [
        pytest.param("script", marks=pytest.mark.xfail(reason="No pytorch impl")),
        pytest.param("trace", marks=pytest.mark.xfail(reason="No pytorch impl")),
        "none",
    ],
)
def test_pytorch_short_integration_frame_computer(include_energy, jit_type):
    torch.manual_seed(5)
    signal = torch.randn(16_000)
    json_ = dict(
        name="si",
        bank="fbank",
        include_energy=include_energy,
    )
    computer_numpy = alias_factory_subclass_from_arg(compute.FrameComputer, json_)
    computer_pytorch = PyTorchSIFrameComputer.from_si_frame_computer(computer_numpy)
    if jit_type == "script":
        computer_pytorch = torch.jit.script(computer_pytorch)
    elif jit_type == "trace":
        with pytest.warns(torch.jit.TracerWarning):
            computer_pytorch = torch.jit.trace(computer_pytorch, (torch.empty(1),))
    exp = computer_numpy.compute_full(signal.numpy())
    with torch.no_grad():
        act = computer_pytorch(signal).numpy()
    assert np.allclose(exp, act, atol=1e-5)
