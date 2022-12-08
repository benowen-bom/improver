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
"""Tests for the apply-bias-correction CLI."""

import pytest

from . import acceptance as acc

pytestmark = [pytest.mark.acc, acc.skip_if_kgo_missing]
CLI = acc.cli_name_with_dashes(__file__)
run_cli = acc.run_cli(CLI)


def test_single_bias_file(tmp_path):
    """
    Test case where bias values are stored in a single file (mean value over
    multiple historic forecasts).
    """
    kgo_dir = acc.kgo_root() / "apply-bias-correction"
    kgo_path = kgo_dir / "single_bias_file" / "kgo.nc"
    fcst_path = kgo_dir / "20220814T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    bias_file_path = (
        kgo_dir / "single_bias_file" / "20220813T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    )
    output_path = tmp_path / "output.nc"
    args = [
        fcst_path,
        bias_file_path,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_multiple_bias_files(tmp_path):
    """
    Test case where bias values are stored over multiple files (single file
    per historic forecasts).
    """
    kgo_dir = acc.kgo_root() / "apply-bias-correction"
    kgo_path = kgo_dir / "multiple_bias_files" / "kgo.nc"
    fcst_path = kgo_dir / "20220814T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    bias_file_paths = (kgo_dir / "multiple_bias_files").glob(
        "202208*T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    )
    output_path = tmp_path / "output.nc"
    args = [
        fcst_path,
        *bias_file_paths,
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)


def test_lower_bound(tmp_path):
    """
    Test case where lower bound is supplied.
    """
    kgo_dir = acc.kgo_root() / "apply-bias-correction"
    kgo_path = kgo_dir / "with_lower_bound" / "kgo.nc"
    fcst_path = kgo_dir / "20220814T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    bias_file_path = (
        kgo_dir / "single_bias_file" / "20220813T0300Z-PT0003H00M-wind_speed_at_10m.nc"
    )
    output_path = tmp_path / "output.nc"
    args = [
        fcst_path,
        bias_file_path,
        "--lower-bound",
        "0",
        "--output",
        output_path,
    ]
    run_cli(args)
    acc.compare(output_path, kgo_path)
