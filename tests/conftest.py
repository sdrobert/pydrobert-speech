# pylint: skip-file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from tempfile import NamedTemporaryFile

import numpy as np
import pytest

from pydrobert.signal import compute
from pydrobert.signal import filters
from pydrobert.signal import scales

# throws away the user error on import if LC_ALL=C is not set
try:
    import pydrobert.kaldi
except ImportError:
    pass

warnings.simplefilter('error')

@pytest.fixture
def temp_file_1_name():
    temp = NamedTemporaryFile(delete=False)
    temp.close()
    yield temp.name
    os.remove(temp.name)

@pytest.fixture(params=[
    scales.LinearScaling(np.random.randint(1, 20)),
    scales.OctaveScaling(np.random.randint(1, 20)),
    scales.MelScaling(),
    scales.BarkScaling(),
], ids=[
    'linear',
    'octave',
    'mel',
    'bark',
], scope='module',
)
def scaling_function(request):
    return request.param

@pytest.fixture(params=[
    1,
    11,
    40,
], ids=[
    '1 filt',
    '11 filts',
    '40 filts',
], scope='module',)
def num_filts(request):
    return request.param

@pytest.fixture(params=[
    lambda scale, num_filts: filters.TriangularOverlappingFilterBank(
        scale,
        low_hz=np.random.randint(20, 60),
        num_filts=num_filts,
        high_hz=np.random.randint(2000, 4000) if np.random.randint(2) else None,
        sampling_rate=8000 if np.random.randint(2) else 16000,
        analytic=np.random.randint(2),
    ),
    lambda scale, num_filts: filters.GaborFilterBank(
        scale,
        low_hz=np.random.randint(20, 60),
        num_filts=num_filts,
        high_hz=np.random.randint(2000, 4000) if np.random.randint(2) else None,
        sampling_rate=8000 if np.random.randint(2) else 16000,
        boundary_adjustment_mode='edges' if np.random.randint(2) else 'wrap',
    ),
    # lambda scale, num_filts: filters.ComplexGammatoneFilterBank(
    #     scale,
    #     low_hz=np.random.randint(20, 60),
    #     num_filts=num_filts,
    #     high_hz=np.random.randint(2000, 4000) if np.random.randint(2) else None,
    #     sampling_rate=8000 if np.random.randint(2) else 16000,
    #     order=np.random.randint(2, 5),
    #     max_centered=np.random.randint(2),
    # ),
], ids=[
    'triangular',
    'gabor',
    # 'gammatone',
])
def bank(request, scaling_function, num_filts):
    return request.param(scaling_function, num_filts)

@pytest.fixture(params=[
    'causal',
    'centered',
], scope='module',)
def frame_style(request):
    return request.param

@pytest.fixture(params=[
    lambda bank, frame_style: compute.STFTFrameComputer(
        bank,
        frame_length_ms=np.random.randint(5, 30) if np.random.randint(2) else None,
        frame_shift_ms=np.random.randint(5, 20),
        use_power=np.random.randint(2),
        use_log=np.random.randint(2),
        pad_to_nearest_power_of_two=np.random.randint(2),
        include_energy=np.random.randint(2),
        frame_style=frame_style
    ),
    lambda bank, frame_style: compute.SIFrameComputer(
        bank,
        frame_shift_ms=np.random.randint(5, 20),
        use_power=np.random.randint(2),
        use_log=np.random.randint(2),
        pad_to_nearest_power_of_two=np.random.randint(2),
        include_energy=np.random.randint(2),
        frame_style=frame_style,
        frequency_smoothing_pre=np.random.choice(('db2', 'tri3', None)),
    ),
], ids=[
    'stft',
    'si',
],)
def computer(request, bank, frame_style):
    return request.param(bank, frame_style)

