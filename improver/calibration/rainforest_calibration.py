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
"""RainForests calibration Plugins."""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from numpy import ndarray
from pandas import DataFrame

from improver import BasePlugin, PostProcessingPlugin
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertProbabilitiesToPercentiles,
)
from improver.ensemble_copula_coupling.utilities import choose_set_of_percentiles
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.cube_manipulation import compare_coords

# Passed to choose_set_of_percentiles to set of evenly spaced percentiles
DEFAULT_ERROR_PERCENTILES_COUNT = 19
DEFAULT_OUTPUT_REALIZATIONS_COUNT = 100


def make_increasing(input_array: ndarray) -> ndarray:
    """Make np.arry monotone increasing in the first dimension.

    Args:
        input_array: the array to make monotone

    Returns:
        array: an array of same shape as the input, where np.diff(axis=0)
        is non-negative
    """
    upper = np.maximum.accumulate(input_array, axis=0)
    lower = np.flip(np.minimum.accumulate(np.flip(input_array, axis=0), axis=0), axis=0)
    return 0.5 * (upper + lower)


def make_decreasing(input_array: ndarray) -> ndarray:
    """Make np.arry monotone decreasing in the first dimension.

    Args:
        input_array: the array to make monotone

    Returns:
        array: an array of same shape as the input, where np.diff(axis=0)
        is non-negative
    """
    lower = np.minimum.accumulate(input_array, axis=0)
    upper = np.flip(np.maximum.accumulate(np.flip(input_array, axis=0), axis=0), axis=0)
    return 0.5 * (upper + lower)


class TrainRainForestsTreeModels(BasePlugin):
    """Class to train tree models for use in RainForests calibration."""


class ApplyRainForestsCalibration(PostProcessingPlugin):
    """Class to calibrate input forecast given a series of RainForests tree models."""

    def __init__(self, model_config_dict: dict, nthreads: int):
        """Initialise the tree model variables used in the application of RainForests
        Calibration.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            nthreads:
                Number of threads to use when initialising tree-model objects.

        Dictionary is of format::

            {
                "-50.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>",
                    "treelite_model" : "<path_to_treelite_model_object>"
                },
                "-25.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>",
                    "treelite_model" : "<path_to_treelite_model_object>"
                },
                ...,
                "50.0" : {
                    "lightgbm_model" : "<path_to_lightgbm_model_object>",
                    "treelite_model" : "<path_to_treelite_model_object>"
                }

        The keys specify the error threshold value, while the associated values
        are the path to the corresponding tree-model objects for that threshold.

        Treelite Predictors are returned if treelite_runtime is an installed dependency and
        an associated path has been supplied for all thresholds, otherwise lightgbm the
        Boosters are returned.

        Returns:
            - numpy array containing the error threshold values.
            - list of tree-model objects to be used in calibration process.
        """
        import importlib

        from lightgbm import Booster

        if importlib.util.find_spec("treelite") is not None:
            from treelite_runtime import Predictor

            self.treelite_enabled = True
        else:
            warnings.warn(
                "Module treelite_runtime unavailable. Defaulting to using lightgbm Boosters."
            )
            self.treelite_enabled = False

        error_thresholds = list(model_config_dict.keys())

        lightgbm_model_filenames = [
            model_config_dict[threshold].get("lightgbm_model")
            for threshold in error_thresholds
        ]
        treelite_model_filenames = [
            model_config_dict[threshold].get("treelite_model")
            for threshold in error_thresholds
        ]
        if (None not in treelite_model_filenames) and self.treelite_enabled:
            self.tree_models = [
                Predictor(libpath=file, verbose=False, nthread=nthreads)
                for file in treelite_model_filenames
            ]
        else:
            self.tree_models = [
                Booster(model_file=file).reset_parameter({"num_threads": nthreads})
                for file in lightgbm_model_filenames
            ]

        self.error_thresholds = np.array(error_thresholds, dtype=np.float32)

    def _check_num_features(self, features: CubeList) -> None:
        """Check that the correct number of features has been passed into the model."""
        sample_tree_model = self.tree_models[0]
        if self.treelite_enabled:
            from treelite_runtime import Predictor

            if isinstance(sample_tree_model, Predictor):
                expected_num_features = sample_tree_model.num_feature
            else:
                expected_num_features = sample_tree_model.num_feature()
        else:
            expected_num_features = sample_tree_model.num_feature()

        if expected_num_features != len(features):
            raise ValueError(
                "Number of expected features does not match number of feature cubes."
            )

    # Does this belong somewhere else?
    def _add_coordinate_to_cube(
        self,
        input_cube: Cube,
        new_coord: DimCoord,
        new_dim_location: Optional[int] = None,
        copy_metadata: Optional[bool] = True,
    ) -> Cube:
        """Create a copy of input cube with an additional dimension coordinate
        added to the cube at the specified axis. The data from input cube is broadcast
        over this new dimension.
        Args:
            input cube:
                cube to add realization dimension to.
            new_coord:
                new coordinate to add to input cube.
            new_dim_location:
                position in cube.data to position the new dimension coord.
            copy_metadata:
                flag as to whether to carry metadata over to output cube.

        Returns:
            output_cube
        """
        input_dim_count = len(input_cube.dim_coords)

        new_dim_coords = list(input_cube.dim_coords) + [new_coord]
        new_dims = list(range(input_dim_count + 1))
        new_dim_coords_and_dims = list(zip(new_dim_coords, new_dims))

        aux_coords = input_cube.aux_coords
        aux_coord_dims = [input_cube.coord_dims(coord.name()) for coord in aux_coords]
        new_aux_coords_and_dims = list(zip(aux_coords, aux_coord_dims))

        new_coord_size = len(new_coord.points)
        new_data = np.broadcast_to(
            input_cube.data, shape=(new_coord_size,) + input_cube.shape
        )
        new_data = input_cube.data[..., np.newaxis] * np.ones(
            shape=new_coord_size, dtype=input_cube.dtype
        )

        output_cube = Cube(
            new_data,
            dim_coords_and_dims=new_dim_coords_and_dims,
            aux_coords_and_dims=new_aux_coords_and_dims,
        )
        if copy_metadata:
            output_cube.metadata = input_cube.metadata

        if new_dim_location is not None:
            final_dim_order = np.insert(
                np.arange(input_dim_count), new_dim_location, values=input_dim_count
            )
            output_cube.transpose(final_dim_order)

        return output_cube

    def _align_feature_variables(
        self, feature_cubes: CubeList, forecast_cube: Cube
    ) -> Tuple[CubeList, Cube]:
        """Ensure that feature cubes have consistent dimension coordinates. If
        realization dimension present in any cube, all cubes lacking this dimension
        will have realization dimension added and broadcast along this new dimension.

        Args:
            feature_cubes:
                cube list containing feature variables to align.
            forecast_cube:
                cube containing the forecast variable to align.

        Returns:
            - feature_cubes with realization coordinate added to each cube if absent
            - forecast_cube with realization coordinate added if absent

        Raises:
            ValueError:
                if feature/forecast variables have inconsistent dimension coordinates
                (excluding realization dimension), or if feature/forecast variables have
                different length realization coordinate over cubes containing this coordinate.
        """
        combined_cubes = CubeList(list([*feature_cubes, forecast_cube]))

        # Compare feature cube coordinates, raise error if dim-coords don't match
        compare_feature_coords = compare_coords(
            combined_cubes, ignored_coords=["realization"]
        )
        misaligned_dim_coords = [
            coord_info["coord"]
            for misaligned_coords in compare_feature_coords
            for coord, coord_info in misaligned_coords.items()
            if coord_info["data_dims"] is not None
        ]
        if misaligned_dim_coords:
            raise ValueError(
                f"Dimension coords do not match between: {misaligned_dim_coords}"
            )

        # Compare realization coordinates across cubes where present;
        # raise error if realization coordinates don't match, otherwise set
        # common_realization_coord to broadcast over.
        realization_coords = {
            variable.name(): variable.coords("realization")
            for variable in combined_cubes
            if variable.coords("realization")
        }
        if not realization_coords:
            # Case I: realization_coords is empty. Add single realization dim to all cubes.
            common_realization_coord = DimCoord(
                [0], standard_name="realization", units=1, var_name="realization"
            )
        else:
            # Case II: realization_coords is not empty.
            variables_with_realization = list(realization_coords.keys())
            sample_variable = variables_with_realization[0]
            sample_realization = realization_coords[sample_variable][0]
            misaligned_realizations = [
                feature
                for feature in variables_with_realization[1:]
                if realization_coords[feature][0] != sample_realization
            ]
            if misaligned_realizations:
                misaligned_realizations.append(sample_variable)
                raise ValueError(
                    f"Realization coords  do not match between: {misaligned_realizations}"
                )
            common_realization_coord = sample_realization

        # Add realization coord to cubes where absent by broadcasting along this dimension
        aligned_cubes = CubeList()
        for i_cube, cube in enumerate(combined_cubes):
            if not cube.coords("realization"):
                cube = combined_cubes[i_cube]
                expanded_cube = self._add_coordinate_to_cube(
                    cube, common_realization_coord, new_dim_location=0
                )
                aligned_cubes.append(expanded_cube)
            else:
                aligned_cubes.append(combined_cubes[i_cube])

        return aligned_cubes[:-1], aligned_cubes[-1]

    def _prepare_error_probability_cube(self, forecast_cube):
        """Initialise a cube with the same dimensions as the input forecast_cube,
        with an additional threshold dimension added as the leading dimension.

        Args:
            forecast_cube:
                Cube containing the forecast to be calibrated.
            error_thresholds:
                Error thresholds corresponding to at which error probabilities are
                to be evaluated using the tree models.

        Returns:
            An empty probability cube.
        """
        # Create a template for error CDF, with threshold the leading dimension.
        error_probability_cube = create_new_diagnostic_cube(
            name=f"probability_of_forecast_error_of_{forecast_cube.name()}_above_threshold",
            units="1",
            template_cube=forecast_cube,
            mandatory_attributes=generate_mandatory_attributes([forecast_cube]),
        )
        threshold_coord = DimCoord(
            self.error_thresholds,
            long_name="threshold",
            units=forecast_cube.units,
            attributes={"spp__relative_to_threshold": "above"},
        )
        error_probability_cube = self._add_coordinate_to_cube(
            error_probability_cube,
            threshold_coord,
            new_dim_location=0,
            copy_metadata=True,
        )

        return error_probability_cube

    def _prepare_features_dataframe(self, feature_cubes: Cube) -> DataFrame:
        """Convert gridded feature cubes into a dataframe, with feature variables
        sorted alphabettically.

        Args:
            feature_cubes:
                Cubelist containing the independent feature variables for prediction.

        Returns:
            Dataframe containing flattened feature variables.

        Raises:
            ValueError:
                If flattened cubes have differing length.
        """
        # Get the names of features and sort alphabetically
        feature_variables = [cube.name() for cube in feature_cubes]
        feature_variables.sort()

        # Unpack the cube-data into dataframe to feed into the tree-models.
        features_df = pd.DataFrame()
        for feature in feature_variables:
            cube = feature_cubes.extract_cube(feature)
            print(cube.name(), cube.units)
            print(f"Flattening: {cube.name()}")
            data = cube.data.flatten()
            if (len(features_df) > 0) and (len(data) != len(features_df)):
                raise RuntimeError("Input cubes have differing sizes.")
            features_df[feature] = data

        print(f"Cube -> dataframe complete: \n{features_df.dtypes}")

        return features_df

    def _get_error_probabilities(
        self,
        forecast_cube: Cube,
        feature_cubes: CubeList,
    ) -> Cube:
        """Evaluate the error exceedence probabilities for forecast_cube from tree_models and
        the associated feature_cubes.

        Args:
            forecast_cube:
                Cube containing the variable to be calibrated.
            feature_cubes:
                Cubelist containing the independent feature variables for prediction.

        Returns:
            A cube containing error exceedence probabilities.

        Raises:
            ValueError:
                If an unsupported model object is passed. Expects lightgbm Booster, or
                treelite_runtime Predictor (if treelite dependency is available).
        """
        from lightgbm import Booster
        if self.treelite_enabled:
            from treelite_runtime import DMatrix, Predictor

        error_probability_cube = self._prepare_error_probability_cube(
            forecast_cube, self.error_thresholds
        )

        features_df = self._prepare_features_dataframe(feature_cubes)

        if isinstance(self.tree_models[0], Booster):
            print("Using light-gbm model:")
            # Use GBDT models for calculation.
            input_dataset = features_df
        elif self.treelite_enabled:
            if isinstance(self.tree_models[0], Predictor):
                print("Using treelite model:")
                # Use treelite models for calculation.
                input_dataset = DMatrix(features_df.values)
            else:
                raise ValueError("Unsupported model object passed.")
        else:
            raise ValueError("Unsupported model object passed.")

        for threshold_index, model in enumerate(self.tree_models):
            print(
                f"Calculating Pr(error > {self.error_thresholds[threshold_index]:0.4f} mm)"
            )
            prediction = model.predict(input_dataset)
            error_probability_cube.data[threshold_index, ...] = np.reshape(
                prediction, forecast_cube.data.shape
            )
            print(f"min: {prediction.min()}, max: {prediction.max()}")

        print("Enforcing monotonicity.")
        error_probability_cube.data = make_decreasing(error_probability_cube.data)

        return error_probability_cube

    def _extract_error_percentiles(
        self, error_probability_cube, error_percentiles_count
    ):
        """Extract error percentile values from the error exceedence probabilities.

        Args:
            error_probability_cube:

            error_percentiles_count:
                The number of error percentiles to extract. The resulting percentiles
                will be evenly spaced on the interval (0, 100).

        Returns:

        """
        error_percentiles = choose_set_of_percentiles(
            error_percentiles_count, sampling="quantile",
        )
        error_percentiles_cube = ConvertProbabilitiesToPercentiles().process(
            error_probability_cube, percentiles=error_percentiles
        )

        return error_percentiles_cube

    def process(
        self,
        forecast_cube: Cube,
        feature_cubes: CubeList,
        error_percentiles_count: int = DEFAULT_ERROR_PERCENTILES_COUNT,
        output_realizations_count: int = DEFAULT_OUTPUT_REALIZATIONS_COUNT,
    ) -> Cube:
        """Apply rainforests calibration to forecast cube.

        The calibration is done in a situation dependent fashion using a series of
        decision-tree models to construct representative error distributions which are
        then used to map each input ensemble member onto a series of realisable values.

        These error distributions are formed in a two-step process:

        1. Evalute error CDF defined over the specified error_thresholds. Each exceedence
        probability is evaluated using the corresponding decision-tree model.

        2. Interpolate the CDF to extract a series of percentiles for the error distribution.
        The error percentiles are then applied to each associated ensemble realization to
        produce a series of realisable values; collectively these series form a calibrated
        super-ensemble, which is then sub-sampled to provide the calibrated forecast.

        Args:
            forecast_cube:
                Cube containing the forecast to be calibrated.
            feature_cubes:
                Cubelist containing the feature variables for prediction.
            error_percentiles_count:
                The number of error percentiles to extract from the associated error CDFs
                evaluated via the tree-models. These error percentiles are applied to each
                ensemble realization to produce a series of values, which collectively form
                the super-ensemble. The resulting super-ensemble will be of
                size = forecast.realization.size * error_percentiles_count.
            output_realizations_count:
                The number of ensemble realizations that will be extracted from the
                super-ensemble. If realizations_count is None, all realizations will
                be returned.

        Returns:
            The calibrated forecast cube.
        """
        # Check that tree-model object available for each error threshold.
        if len(self.error_thresholds) != len(self.tree_models):
            raise ValueError(
                "tree_models must be of the same size as error_thresholds."
            )

        # Check that the correct number of feature variables has been supplied.
        self._check_num_features(feature_cubes)

        # Align forecast and feature datasets
        aligned_features, aligned_forecast = self._align_feature_variables(
            feature_cubes, forecast_cube
        )

        # Evaluate the error CDF using tree-models.
        error_CDF = self._get_error_probabilities(
            aligned_forecast, aligned_features
        )

        # Extract error percentiles from error CDF.
        error_percentiles = self._extract_error_percentiles(
            error_CDF, error_percentiles_count
        )
        error_percentiles

        # Apply error to forecast cube.

        # Combine sub-ensembles into a single consolidated ensemble.
