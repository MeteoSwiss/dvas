# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

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

    #Now something more specific, to make sure I can use these function also for a groupby() entity.
    assert np.isnan(vals[0].groupby(vals.index//2).aggregate(tools.fancy_nansum, axis=0)[0])
    assert np.all(vals[0].groupby(vals.index//2).aggregate(tools.fancy_nansum, axis=0)[1:] ==
                  [1, 1])

def test_weighted_mean():
    """ Function used to test the weighted_mean combination of profiles.

    """

    vals = pd.DataFrame(np.ones((10, 3)))
    vals[1] = 2.
    vals[2] = 4.
    weights = pd.DataFrame(np.ones((10, 3)))

    # Can I actually compute a weighted mean ?
    assert np.all(tools.weighted_mean(vals, weights, binning=1)[0] == 7/3)
    assert np.all(tools.weighted_mean(vals, weights, binning=3)[0] == 7/3)

    vals.iloc[0] = np.nan
    vals[0][1] = np.nan
    weights[1][1] = np.nan
    weights.iloc[9] = np.nan

    # If all values for a bin a NaN, result should be NaN
    assert np.isnan(tools.weighted_mean(vals, weights, binning=1)[0][0])
    # But if only some of the bin are NaNs, I should return a number
    assert tools.weighted_mean(vals, weights, binning=1)[0][1] == 4.0
    assert tools.weighted_mean(vals, weights, binning=2)[0][0] == 4.0
    assert tools.weighted_mean(vals, weights, binning=2)[0][1] == 7/3

    # If all weights for a bin are NaN, result should be Nan
    assert np.isnan(tools.weighted_mean(vals, weights, binning=1)[0][9])

    # Check that the last bin is smaller than the others, I still compute it
    assert len(tools.weighted_mean(vals, weights, binning=3)[0]) == 4

def test_delta():
    """ Function used to test the weighted_mean combination of profiles.

    """

    vals = pd.DataFrame(np.ones((10, 2)))
    vals[1] = 2.

    # Can I actually compute a difference ?
    assert np.all(tools.delta(vals, binning=1)[0] == -1)

    vals.iloc[0] = np.nan
    vals[0][9] = np.nan

    # If all values for a bin a NaN, result should be NaN
    assert np.isnan(tools.delta(vals, binning=1)[0][0])

    # If only part of a bin is NaN, then report a number
    assert tools.delta(vals, binning=2)[0][0] == -1

    # If only on number is NaN, result should be Nan
    assert np.isnan(tools.delta(vals, binning=1)[0][9])

    # Check that the last bin is smaller than the others, I still compute it
    assert len(tools.delta(vals, binning=4)[0]) == 3
