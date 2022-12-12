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

import iris
import numpy as np
import numpy.ma as ma
import pytest
from iris.cube import Cube, CubeList

from improver.calibration.simple_bias_correction import (
    ApplyBiasCorrection,
    apply_additive_correction,
)
from improver.calibration.utilities import create_unified_frt_coord
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import collapsed, get_dim_coord_names

VALID_TIME = datetime(2022, 12, 6, 3, 0)

ATTRIBUTES = {
    "title": "Test forecast dataset",
    "model_configuration": "fcst_model",
    "source": "IMPROVER",
    "institution": "Australian Bureau of Meteorology",
}

RNG = np.random.default_rng(0)

TEST_FCST_DATA = np.array(
    [[1.0, 2.0, 2.0], [2.0, 1.0, 3.0], [1.0, 3.0, 3.0]], dtype=np.float32
) + RNG.normal(0.0, 1, (4, 3, 3)).astype(np.float32)


MEAN_BIAS_DATA = np.array(
    [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-2.0, 0.0, 1.0]], dtype=np.float32
)

MASK = np.array(
    [[False, False, False], [True, False, False], [True, False, True]], dtype=bool
)


@pytest.fixture
def forecast_cube():
    return set_up_variable_cube(
        data=TEST_FCST_DATA,
        name="wind_speed",
        units="m/s",
        time=VALID_TIME + timedelta(days=1),
        frt=VALID_TIME + timedelta(days=1) - timedelta(hours=3),
        attributes=ATTRIBUTES,
    )


def generate_bias_cubelist(
    num_frts: int,
    single_frt_with_bounds: bool = False,
    last_valid_time: datetime = VALID_TIME,
    masked_data: bool = False,
):
    """Generate sample bias CubeList for testing.

    Args:
        num_frts:
            Number of forecast_reference_times to use in constructing the bias values.
        single_frt_with_bounds:
            Flag to specify whether to return a single (mean) bias value defined over
            the set of frt values, or leave as multiple bias values defined over a single
            frt.
        last_valid_time:
            The latest valid time to use in constructing the set of bias values. All
            associated valid-times (and frts) are evaluated by stepping back day-by-day
            num_frt times.
        masked_data:
            Flag as to whether to mask a portion of the bias data.

    Returns:
        CubeList containg the sample bias cubes.
    """
    data = MEAN_BIAS_DATA
    # Set the attributes to those expected for bias cube.
    attributes = ATTRIBUTES.copy()
    del attributes["model_configuration"]
    attributes["title"] = "Forecast bias data"

    # Initialise random number generator for adding noise around MEAN_BIAS_DATA.
    rng = np.random.default_rng(0)

    bias_cubes = CubeList()
    for i in range(num_frts):
        if num_frts > 1:
            # Add noise when constructing multiple values
            noise = rng.normal(0.0, 0.1, (3, 3)).astype(np.float32)
            data_slice = data + noise
        else:
            data_slice = data
        # Apply mask to data if using masked bias dataset
        if masked_data:
            data_slice = ma.asarray(data_slice, dtype=data_slice.dtype)
            data_slice.mask = MASK

        bias_cube = set_up_variable_cube(
            data=data_slice,
            name="wind_speed_forecast_error",
            units="m/s",
            time=last_valid_time + timedelta(hours=3) - timedelta(days=i),
            frt=last_valid_time - timedelta(days=i),
            attributes=attributes,
        )
        bias_cube.remove_coord("time")
        bias_cubes.append(bias_cube)

    # Collapse down frt coord if using mean bias value defined in single cube.
    if single_frt_with_bounds and num_frts > 1:
        bias_cubes = bias_cubes.merge_cube()
        frt_coord = create_unified_frt_coord(
            bias_cubes.coord("forecast_reference_time")
        )
        bias_cubes = collapsed(
            bias_cubes, "forecast_reference_time", iris.analysis.MEAN
        )
        bias_cubes.data = bias_cubes.data.astype(bias_cubes.dtype)
        bias_cubes.replace_coord(frt_coord)
        bias_cubes = CubeList([bias_cubes])

    return bias_cubes


@pytest.mark.parametrize("num_bias_inputs", (1, 30))
@pytest.mark.parametrize("masked_bias_data", (True, False))
@pytest.mark.parametrize("fill_masked_bias_data", (True, False))
def test_apply_additive_correction(
    forecast_cube, num_bias_inputs, masked_bias_data, fill_masked_bias_data
):
    """Test the additive correction provides expected value."""
    bias_cube = generate_bias_cubelist(
        num_bias_inputs, single_frt_with_bounds=True, masked_data=masked_bias_data
    )[0]

    expected = TEST_FCST_DATA - MEAN_BIAS_DATA
    if fill_masked_bias_data and masked_bias_data:
        expected = np.where(MASK, TEST_FCST_DATA, expected)

    result = apply_additive_correction(forecast_cube, bias_cube, fill_masked_bias_data)

    if masked_bias_data and not fill_masked_bias_data:
        assert isinstance(result, ma.masked_array)
        assert np.all(result.mask == MASK)
        assert np.ma.allclose(result, expected, atol=0.05)
    else:
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, expected, atol=0.05)


def test__init__():
    """Test that the class functions are set to the expected values."""
    plugin = ApplyBiasCorrection()
    assert plugin.correction_method == apply_additive_correction


@pytest.mark.parametrize("single_input_frt", (False, True))
def test_get_mean_bias(single_input_frt):
    """Test that mean value cube returned has expected properties."""
    input_cubelist = generate_bias_cubelist(30, single_frt_with_bounds=single_input_frt)
    result = ApplyBiasCorrection()._get_mean_bias(input_cubelist)

    # Check that the CubeList has been collapsed down to a single value along
    # the forecast_reference_time coord.
    assert "forecast_reference_time" not in get_dim_coord_names(result)
    # Check that the resultant value is the expected mean value (within tolerance).
    assert np.allclose(result.data, MEAN_BIAS_DATA, atol=0.05)
    # Check that the return type is an iris.cube.Cube
    assert isinstance(result, Cube)
    # Check conistent datatype
    assert result.dtype == input_cubelist[0].dtype


def test_get_mean_bias_fails_on_inconsistent_bounds():
    """Test that get_mean_bias fails when passing in multiple bias values defined
    over a range of forecast_reference_times."""
    # Set up cube inputs defined over multiple frt values.
    input_cubelist = CubeList()
    input_cubelist.extend(generate_bias_cubelist(2, single_frt_with_bounds=True))
    input_cubelist.extend(
        generate_bias_cubelist(
            2,
            single_frt_with_bounds=True,
            last_valid_time=VALID_TIME - timedelta(days=2),
        )
    )
    with pytest.raises(ValueError):
        ApplyBiasCorrection()._get_mean_bias(input_cubelist)
    # Set up cube inputs with mixed definition: one over single frt, the other over
    # multiple frt values.
    input_cubelist = CubeList()
    input_cubelist.extend(generate_bias_cubelist(2, single_frt_with_bounds=False))
    input_cubelist.extend(
        generate_bias_cubelist(
            2,
            single_frt_with_bounds=True,
            last_valid_time=VALID_TIME - timedelta(days=2),
        )
    )
    with pytest.raises(ValueError):
        ApplyBiasCorrection()._get_mean_bias(input_cubelist)


@pytest.mark.parametrize("num_bias_inputs", (1, 30))
@pytest.mark.parametrize("single_input_frt", (False, True))
@pytest.mark.parametrize("lower_bound", (None, 1))
@pytest.mark.parametrize("masked_input_data", (True, False))
@pytest.mark.parametrize("masked_bias_data", (True, False))
@pytest.mark.parametrize("fill_masked_bias_data", (True, False))
def test_process(
    forecast_cube,
    num_bias_inputs,
    single_input_frt,
    lower_bound,
    masked_input_data,
    masked_bias_data,
    fill_masked_bias_data,
):
    """Test process function over range of input types, with/without lower bound."""
    input_bias_cubelist = generate_bias_cubelist(
        num_bias_inputs,
        single_frt_with_bounds=single_input_frt,
        masked_data=masked_bias_data,
    )
    if masked_input_data:
        forecast_cube.data = ma.asarray(
            forecast_cube.data, dtype=forecast_cube.data.dtype
        )
        forecast_cube.data.mask = MASK
    result = ApplyBiasCorrection().process(
        forecast_cube, input_bias_cubelist, lower_bound, fill_masked_bias_data
    )
    expected = TEST_FCST_DATA - MEAN_BIAS_DATA
    if fill_masked_bias_data and masked_bias_data:
        expected = np.where(MASK, TEST_FCST_DATA, expected)
    if lower_bound is not None:
        expected = np.maximum(lower_bound, expected)
    # Check values are as expected (within tolerance)
    assert np.ma.allclose(result.data, expected, atol=0.05)
    # Check the cube.data type is as expected based on input forecast
    # and bias_values. Here we are checking masked values are handled as expected.
    if (masked_bias_data and not fill_masked_bias_data) or masked_input_data:
        assert isinstance(result.data, ma.masked_array)
    else:
        assert isinstance(result.data, np.ndarray)
    # Check the dtypes match
    assert result.dtype == forecast_cube.dtype
    # Check variable metadata is consistent
    assert result.standard_name == forecast_cube.standard_name
    assert result.long_name == forecast_cube.long_name
    assert result.var_name == forecast_cube.var_name
    assert result.units == forecast_cube.units
    # Check coords and attributes are consistent
    assert result.coords() == forecast_cube.coords()
    assert result.attributes == forecast_cube.attributes
