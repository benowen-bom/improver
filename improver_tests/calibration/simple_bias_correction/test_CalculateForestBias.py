# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from datetime import datetime, timedelta

import numpy as np
import pytest
from iris.cube import CubeList

from improver.calibration.simple_bias_correction import CalculateForecastBias
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import get_coord_names, get_dim_coord_names

ATTRIBUTES = {
    "title": "Test forecast dataset",
    "model_configuration": "fcst_model",
    "source": "IMPROVER",
    "institution": "Australian Bureau of Meteorology",
}

VALID_TIME = datetime(2022, 12, 6, 3, 0)


def generate_dataset(num_frt=1, truth_dataset=False):

    attributes = ATTRIBUTES.copy()

    times = [VALID_TIME - i * timedelta(days=1) for i in range(num_frt)]
    if truth_dataset:
        period = timedelta(hours=0)
        attributes["title"] = "Test truth dataset"
        attributes["model_configuration"] = "truth_data"
    else:
        period = timedelta(hours=3)
    forecast_ref_times = {time: time - period for time in times}

    rng = np.random.default_rng(0)
    data_shape = (10, 10)

    ref_forecast_cubes = CubeList()
    for time in times:
        ref_forecast_cubes.append(
            set_up_variable_cube(
                np.maximum(0, rng.normal(0.002, 0.001, data_shape)).astype(np.float32),
                time=time,
                frt=forecast_ref_times[time],
                attributes=attributes,
            )
        )
    ref_forecast_cube = ref_forecast_cubes.merge_cube()

    return ref_forecast_cube


# Test case where we have a single or multiple reference forecasts.
@pytest.mark.parametrize("num_frt", (1, 4))
def test__define_metadata(num_frt):

    reference_forecast_cubes = generate_dataset(num_frt)

    expected = ATTRIBUTES.copy()
    expected["title"] = "Forecast bias data"
    # Don't expect this attribute to be carried over to forecast bias data.
    del expected["model_configuration"]

    actual = CalculateForecastBias()._define_metadata(reference_forecast_cubes)

    assert actual == expected


# Test case where we have a single or multiple reference forecasts.
@pytest.mark.parametrize("num_frt", (1, 4))
def test__create_bias_cube(num_frt):

    reference_forecast_cubes = generate_dataset(num_frt)
    result = CalculateForecastBias()._create_bias_cube(reference_forecast_cubes)

    # Check all but the time dim coords are consistent
    expected_dim_coords = set(get_dim_coord_names(reference_forecast_cubes))
    actual_dim_coords = set(get_dim_coord_names(result))
    if num_frt > 1:
        assert expected_dim_coords - actual_dim_coords == set(["time"])
    else:
        assert actual_dim_coords == expected_dim_coords

    # dtypes are consistent
    assert reference_forecast_cubes.dtype == result.dtype

    # Check that frt coord
    if num_frt > 1:
        assert (
            result.coord("forecast_reference_time").points
            == reference_forecast_cubes.coord("forecast_reference_time").points[-1]
        )
        assert np.all(
            result.coord("forecast_reference_time").bounds
            == [
                reference_forecast_cubes.coord("forecast_reference_time").points[0],
                reference_forecast_cubes.coord("forecast_reference_time").points[-1],
            ]
        )
    else:
        assert result.coord(
            "forecast_reference_time"
        ) == reference_forecast_cubes.coord("forecast_reference_time")

    # Check that time coord has been removed
    assert "time" not in get_coord_names(result)

    # Check variable name is as expected
    assert result.long_name == f"{reference_forecast_cubes.name()} forecast error"
