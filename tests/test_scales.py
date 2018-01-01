import numpy as np
import pytest

from pydrobert.speech import scales


@pytest.fixture(params=[
    scales.LinearScaling(10),
    scales.OctaveScaling(19),
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


def test_scales_invertible(scaling_function):
    for hertz in range(20, 8000):
        scale = scaling_function.hertz_to_scale(hertz)
        assert np.isclose(hertz, scaling_function.scale_to_hertz(scale)), (
            "Inverse not equal to orig for {} at {}".format(
                scaling_function,
                hertz,
            )
        )
