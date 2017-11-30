# pylint: skip-file

import numpy as np


def test_scales_invertible(scaling_function):
    for hertz in range(20, 8000):
        scale = scaling_function.hertz_to_scale(hertz)
        assert np.isclose(hertz, scaling_function.scale_to_hertz(scale)), (
            "Inverse not equal to orig for {} at {}".format(
                scaling_function,
                hertz,
            )
        )
