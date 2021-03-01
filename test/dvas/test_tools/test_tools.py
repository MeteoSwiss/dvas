# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

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
