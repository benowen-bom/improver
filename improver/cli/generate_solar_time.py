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
"""Script to run GenerateSolarTime ancillary generation."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    target_grid: cli.inputcube, *, time: cli.inputdatetime, new_title: str = None
):
    """Generate a cube containing local solar time, evaluated on the target grid for
    specified time. Local solar time is used as an input to the RainForests calibration
    for rainfall.

    Args:
        target_grid (iris.cube.Cube):
            A cube with the desired grid.
        time (str):
            A datetime specified in the format YYYYMMDDTHHMMZ at which to calculate the
            local solar time.
        new_title:
            New title for the output cube attributes. If None, this attribute is left out
            since it has no prescribed standard.

    Returns:
        iris.cube.Cube:
            A cube containing local solar time.
    """
    from improver.generate_ancillaries.generate_derived_solar_fields import (
        GenerateSolarTime,
    )

    return GenerateSolarTime()(target_grid, time, new_title=new_title)
