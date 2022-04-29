# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Unit tests for the ApplyRainForestsCalibration class."""
import importlib
import sys

import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import CubeList

from improver.calibration.rainforest_calibration import ApplyRainForestsCalibration
from improver.synthetic_data.set_up_test_cubes import set_up_variable_cube
from improver.utilities.cube_manipulation import compare_attributes, compare_coords

if importlib.util.find_spec("treelite") is not None:
    import treelite_runtime

    TREELITE_ENABLED = True
else:
    TREELITE_ENABLED = False

lightgbm = pytest.importorskip("lightgbm")

EMPTY_COMPARSION_DICT = [{}, {}]


class MockBooster:
    def __init__(self, model_file, **kwargs):
        self.model_class = "lightgbm-Booster"

    def reset_parameter(self, params):
        self.n_threads = params.get("num_threads")
        return self


class MockPredictor:
    def __init__(self, libpath, nthread, **kwargs):
        self.model_class = "treelite-Predictor"
        self.n_threads = nthread


def test__init_lightgbm_models(monkeypatch, lightgbm_model_config, error_thresholds):
    """Test lightgbm models are loaded if model_config contains path to lightgbm models only."""
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    result = ApplyRainForestsCalibration(lightgbm_model_config, nthreads=8)

    for model in result.tree_models:
        assert model.model_class == "lightgbm-Booster"
        assert model.n_threads == 8
    assert result.treelite_enabled == TREELITE_ENABLED
    assert np.all(result.error_thresholds == error_thresholds)


@pytest.mark.skipif(not TREELITE_ENABLED, reason="Required dependency missing.")
def test__init_treelite_models(monkeypatch, treelite_model_config, error_thresholds):
    """Test treelite models are loaded if model_config correctly. If all thresholds
    contain treelite model, treelite Predictor is returned, otherwise return lightgbm
    Boosters."""
    monkeypatch.setattr(treelite_runtime, "Predictor", MockPredictor)
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    result = ApplyRainForestsCalibration(treelite_model_config, nthreads=8)

    for model in result.tree_models:
        assert model.model_class == "treelite-Predictor"
        assert model.n_threads == 8
    assert result.treelite_enabled is True
    assert np.all(result.error_thresholds == error_thresholds)

    # Model type should default to lightgbm if there are any treelite models
    # missing across any thresholds
    treelite_model_config["0.0000"].pop("treelite_model", None)
    result = ApplyRainForestsCalibration(treelite_model_config, nthreads=8)

    for model in result.tree_models:
        assert model.model_class == "lightgbm-Booster"
        assert model.n_threads == 8
    assert result.treelite_enabled is True
    assert np.all(result.error_thresholds == error_thresholds)


def test__init_treelite_missing(monkeypatch, treelite_model_config, error_thresholds):
    """Test that lightgbm Booster returned when model_config references treelite models,
    but treelite dependency is missing."""
    # Simulate environment which does not have treelite loaded.
    monkeypatch.setitem(sys.modules, "treelite", None)
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    result = ApplyRainForestsCalibration(treelite_model_config, nthreads=8)

    for model in result.tree_models:
        assert model.model_class == "lightgbm-Booster"
        assert model.n_threads == 8
    assert result.treelite_enabled is False
    assert np.all(result.error_thresholds == error_thresholds)


def test__align_feature_variables_ensemble(
    monkeypatch, ensemble_features, ensemble_forecast, lightgbm_model_config
):
    """Check cube alignment when using feature and forecast variables when realization
    coordinate present in some cube variables."""

    # For this test we will not be using tree-models, so we will override the
    # initialisation of Class variables.
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    (aligned_features, aligned_forecast,) = ApplyRainForestsCalibration(
        lightgbm_model_config, nthreads=1
    )._align_feature_variables(ensemble_features, ensemble_forecast)

    input_cubes = CubeList([*ensemble_features, ensemble_forecast])
    output_cubes = CubeList([*aligned_features, aligned_forecast])

    # Check that the realization dimension is the outer-most dimension
    assert np.all([cube.coord_dims("realization") == (0,) for cube in output_cubes])

    # Check that all cubes have consistent shape
    assert np.all(
        [cube.data.shape == output_cubes[0].data.shape for cube in output_cubes[1:]]
    )

    # Check the other properties of the cubes are unchanged.
    for input_cube, output_cube in zip(input_cubes, output_cubes):
        assert (
            compare_coords([output_cube, input_cube], ignored_coords="realization")
            == EMPTY_COMPARSION_DICT
        )
        assert compare_attributes([output_cube, input_cube]) == EMPTY_COMPARSION_DICT


def test__align_feature_variables_deterministic(
    monkeypatch, deterministic_features, deterministic_forecast, lightgbm_model_config
):
    """Check cube alignment when using feature and forecast variables when no realization
    coordinate present in any of the cube variables."""
    # For this test we will not be using tree-models, so we will override the
    # initialisation of Class variables.
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    (aligned_features, aligned_forecast,) = ApplyRainForestsCalibration(
        lightgbm_model_config, nthreads=1
    )._align_feature_variables(deterministic_features, deterministic_forecast)

    input_cubes = CubeList([*deterministic_features, deterministic_forecast])
    output_cubes = CubeList([*aligned_features, aligned_forecast])

    # Check that all cubes have realization dimension of length 1
    assert np.all([cube.coord("realization").shape == (1,) for cube in output_cubes])

    # Check that the realization dimension is the outer-most dimension
    assert np.all([cube.coord_dims("realization") == (0,) for cube in output_cubes])

    # Check that all cubes have consistent shape
    assert np.all(
        [cube.data.shape == output_cubes[0].data.shape for cube in output_cubes[1:]]
    )

    # Check the other properties of the cubes are unchanged.
    for input_cube, output_cube in zip(input_cubes, output_cubes):
        assert (
            compare_coords([output_cube, input_cube], ignored_coords="realization")
            == EMPTY_COMPARSION_DICT
        )
        assert compare_attributes([output_cube, input_cube]) == EMPTY_COMPARSION_DICT


def test__align_feature_variables_misaligned_dim_coords(
    monkeypatch, ensemble_features, lightgbm_model_config
):

    # For this test we will not be using tree-models, so we will override the
    # initialisation of Class variables.
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    misaligned_forecast_cube = set_up_variable_cube(
        np.maximum(0, np.random.normal(0.002, 0.001, (5, 10, 15))).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=np.arange(5),
    )

    with pytest.raises(ValueError):
        ApplyRainForestsCalibration(
            lightgbm_model_config, nthreads=1
        )._align_feature_variables(ensemble_features, misaligned_forecast_cube)

    misaligned_forecast_cube = set_up_variable_cube(
        np.maximum(0, np.random.normal(0.002, 0.001, (10, 10, 10))).astype(np.float32),
        name="lwe_thickness_of_precipitation_amount",
        units="m",
        realizations=np.arange(10),
    )

    with pytest.raises(ValueError):
        ApplyRainForestsCalibration(
            lightgbm_model_config, nthreads=1
        )._align_feature_variables(ensemble_features, misaligned_forecast_cube)


@pytest.mark.parametrize(
    "new_dim_location, copy_metadata",
    [(None, True), (None, False), (0, True), (1, True)],
)
def test_add_coordinate(
    monkeypatch,
    deterministic_forecast,
    new_dim_location,
    copy_metadata,
    lightgbm_model_config,
):
    """Test adding dimension to input_cube"""

    # For this test we will not be using tree-models, so we will override the
    # initialisation of Class variables.
    monkeypatch.setattr(lightgbm, "Booster", MockBooster)

    realization_coord = DimCoord(np.arange(0, 5), standard_name="realization", units=1)

    output_cube = ApplyRainForestsCalibration(
        lightgbm_model_config, nthreads=1
    )._add_coordinate_to_cube(
        deterministic_forecast,
        realization_coord,
        new_dim_location=new_dim_location,
        copy_metadata=copy_metadata,
    )

    # Test all but added coord are consistent
    assert (
        compare_coords(
            [output_cube, deterministic_forecast], ignored_coords="realization"
        )
        == EMPTY_COMPARSION_DICT
    )
    if copy_metadata:
        assert (
            compare_attributes([output_cube, deterministic_forecast])
            == EMPTY_COMPARSION_DICT
        )
    else:
        assert (
            compare_attributes([output_cube, deterministic_forecast])
            != EMPTY_COMPARSION_DICT
        )

    # Test realization coord
    output_realization_coord = output_cube.coord("realization")
    assert np.allclose(output_realization_coord.points, realization_coord.points)
    assert output_realization_coord.standard_name == realization_coord.standard_name
    assert output_realization_coord.units == realization_coord.units

    # Test data values
    consistent_data = [
        np.allclose(realization.data, deterministic_forecast.data)
        for realization in output_cube.slices_over("realization")
    ]
    assert np.all(consistent_data)

    # Check dim is in the correct place
    if new_dim_location is None:
        assert output_cube.coord_dims("realization") == (output_cube.ndim - 1,)
    else:
        assert output_cube.coord_dims("realization") == (new_dim_location,)
