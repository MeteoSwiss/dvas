"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.tools.math module.

"""

# Import from python packages and modules
import pytest
import pandas as pd

from dvas.tools.math import calc_tropopause, TropopauseError

def test_calc_tropopause():
    """Test calc_tropopause function

    The test is based on a simulated temperature profile

    """

    # Create temperature profile
    TROPO_ALT = 8000
    TROPO_TEMP = -60
    GROUND_ALT = 1500
    GROUND_TEMP = 0
    UP_SPEED = 5
    time = pd.timedelta_range(
        '0s', f'{round((TROPO_ALT - GROUND_ALT)/UP_SPEED*1.5)}s', freq='2s'
    )
    alt = pd.Series(
        GROUND_ALT + UP_SPEED * time.total_seconds(),  # noqa, pylint: disable=E1101
        index=time
    )
    temp = (alt - GROUND_ALT) * (TROPO_TEMP - GROUND_TEMP)/(TROPO_ALT - GROUND_ALT)
    temp[alt >= TROPO_ALT] = TROPO_TEMP + (alt[alt >= TROPO_ALT] - TROPO_ALT) * 1/10000

    # Calculate
    tropo_time = calc_tropopause(temp, alt)

    # Test
    assert alt.loc[alt.index == tropo_time].values[0] == TROPO_ALT

    # Test exceptions
    with pytest.raises(TropopauseError):
        calc_tropopause(temp, alt, start_alt=alt.max()+1)

    with pytest.raises(TropopauseError):
        calc_tropopause(temp, alt, grad_cond=10)
