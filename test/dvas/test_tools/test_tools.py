# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing 'tools' classes and function of the tools submodule.

"""

# Import from python packages and modules
import numpy as np
import pandas as pd

# Import from this package
from dvas.tools import tools


def test_fancy_nansum():
    """ Function to test if the fancy_nansum works as intended.

    This function test:
      - ability to use it in a vectorial manner
      - ability to use it via .apply() for groupby() entities.
    """

    # Create a fake dataset
    vals = pd.DataFrame(np.ones((5, 3)))
    vals.iloc[0] = np.nan
    vals[0][1] = np.nan
    vals[1][1] = np.nan
    vals[0][2] = np.nan

    # First some basi tests to make sure all works as intended.
    assert tools.fancy_nansum(vals) == 9.0
    assert np.all(tools.fancy_nansum(vals, axis=0).values == [2, 3, 4])
    assert np.isnan(tools.fancy_nansum(vals, axis=1).values[0])
    assert np.all(tools.fancy_nansum(vals, axis=1).values[1:] == [1, 2, 3, 3])

    # Now something more specific, to make sure I can use these function also for a groupby()
    # entity.
    assert np.isnan(vals[0].groupby(vals.index//2).aggregate(tools.fancy_nansum, axis=0)[0])
    assert np.all(vals[0].groupby(vals.index//2).aggregate(tools.fancy_nansum, axis=0)[1:] ==
                  [1, 1])

    # Also make sure it works well with timedelta64[ns] types. See dvas issue #122.
    vals = pd.DataFrame(index=range(10), columns=['tdt'], dtype='timedelta64[ns]')
    assert np.isnan(tools.fancy_nansum(vals))
    assert np.isnan(tools.fancy_nansum(vals, axis=1)).all()


def test_fancy_bitwise_or():
    """ Function to test if the fancy_bitwise_or fct works as expected. """

    # Create a fake dataset
    vals = pd.DataFrame(np.array([[0, 0, 0],
                                  [0, 1, 2],
                                  [1, 3, 4],
                                  [1, 1, 1]])).astype(int)

    out = tools.fancy_bitwise_or(vals, axis=None)
    assert out == 7

    out = tools.fancy_bitwise_or(vals, axis=0)
    assert all(out == pd.array([1, 3, 7]))

    out = tools.fancy_bitwise_or(vals, axis=1)
    assert all(out[1:] == pd.array([3, 7, 1]))

    # Also make sure it works with a single column
    vals = pd.DataFrame(np.array([0, 0, 0])).astype(int)

    out = tools.fancy_bitwise_or(vals, axis=None)
    assert out == 0


def test_wrap_angle():
    """ Function to test the wrap_angle routine """

    assert tools.wrap_angle(np.nan) is np.nan
    assert tools.wrap_angle(None) is None
    assert tools.wrap_angle(18) == 18.
    assert tools.wrap_angle(361) == 1.
    assert tools.wrap_angle(182) == - 178
    assert tools.wrap_angle(-45) == -45
    assert tools.wrap_angle(-720) == 0.
    assert tools.wrap_angle(180) == -180
