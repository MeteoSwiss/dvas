# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing 'gruan' classes and function of the tools submodule.

"""

# Import from python packages and modules
import numpy as np
import pytest

# Function to test
from dvas.tools.gdps.correlations import coeffs

# Define a series of reference measurement parameters, to test the different combination options.
T_OK = [np.zeros(1), np.zeros(1)]                       # Same time
T_NOK = [np.ones(1), np.zeros(1)]                       # Different times
SRN_OK = [np.array(['srn']), np.array(['srn'])]         # Same Serial Number
SRN_NOK = [np.array(['srn a']), np.array(['srn b'])]    # Different SRN
MDL_OK = [np.array(['mod']), np.array(['mod'])]         # Same GDP model
MDL_NOK = [np.array(['mod a']), np.array(['mod b'])]    # Different GDP model
RIG_OK = [np.array(['rig']), np.array(['rig'])]         # Same rig name
RIG_NOK = [np.array(['rig a']), np.array(['rig b'])]    # Different rig names
EVT_OK = [np.array(['evt']), np.array(['evt'])]         # Same event
EVT_NOK = [np.array(['evt x']), np.array(['evt y'])]    # Different event

# Let us test a series of conditions for the different types of uncertainty types
@pytest.mark.parametrize("test_input_1, expected_1", [
    # Uncorrelated errors: same point
    (T_OK + ['ucu'] + SRN_OK + MDL_OK + RIG_OK + EVT_OK, 1),
    # Uncorrelated errors: different time
    (T_NOK + ['ucu'] + SRN_OK + MDL_OK + RIG_OK + EVT_OK, 0),
    # Uncorrelated errors: different Serial Number
    (T_OK + ['ucu'] + SRN_NOK + MDL_OK + RIG_OK + EVT_OK, 0),
    # Uncorrelated errors: different rig
    (T_OK + ['ucu'] + SRN_NOK + MDL_OK + RIG_NOK + EVT_OK, 0),
    # Uncorrelated errors: different events
    (T_OK + ['ucu'] + SRN_NOK + MDL_OK + RIG_OK + EVT_NOK, 0),
    #
    # Rig-correlated errors: same point
    (T_OK + ['ucr'] + SRN_OK + MDL_OK + RIG_OK + EVT_OK, 1),
    # Uncorrelated errors: different time
    (T_NOK + ['ucr'] + SRN_OK + MDL_OK + RIG_OK + EVT_OK, 0),
    # Uncorrelated errors: different Serial Number
    (T_OK + ['ucr'] + SRN_NOK + MDL_OK + RIG_OK + EVT_OK, 0),
    # Uncorrelated errors: different rig
    (T_OK + ['ucr'] + SRN_NOK + MDL_OK + RIG_NOK + EVT_OK, 0),
    # Uncorrelated errors: different events
    (T_OK + ['ucr'] + SRN_NOK + MDL_OK + RIG_OK + EVT_NOK, 0),
    #
    # Spatial-correlated errors: same point
    (T_OK + ['ucs'] + SRN_OK + MDL_OK + RIG_OK + EVT_OK, 1),
    # Spatial-correlated errors: different times
    (T_NOK + ['ucs'] + SRN_NOK + MDL_OK + RIG_OK + EVT_OK, 1),
    # Uncorrelated errors: different rigs
    (T_OK + ['ucs'] + SRN_NOK + MDL_OK + RIG_NOK + EVT_OK, 1),
    # Uncorrelated errors: different rigs  & different times
    (T_NOK + ['ucs'] + SRN_NOK + MDL_OK + RIG_NOK + EVT_OK, 1),
    # Uncorrelated errors: different event
    (T_OK + ['ucs'] + SRN_NOK + MDL_OK + RIG_OK + EVT_NOK, 0),
    #
    # Temporal-correlated errors: same point
    (T_OK + ['uct'] + SRN_OK + MDL_OK + RIG_OK + EVT_OK, 1),
    # Temporal-correlated errors: different time
    (T_NOK + ['uct'] + SRN_NOK + MDL_OK + RIG_OK + EVT_OK, 1),
    # Temporal-correlated errors: different rigs & different times
    (T_NOK + ['uct'] + SRN_NOK + MDL_OK + RIG_NOK + EVT_OK, 1),
    # Temporal-correlated errors: different events and times
    (T_NOK + ['uct'] + SRN_NOK + MDL_OK + RIG_OK + EVT_NOK, 1),
    ])

def test_coeffs(test_input_1, expected_1):
    """Function used to test if the GDP correlations are properly implemented.

    The function tests:
        - correlation coefficients for the different measurement types

    """

    # If I get here, then most likely it is all working fine.
    assert coeffs(test_input_1[0], test_input_1[1], test_input_1[2],
                  oid_i=test_input_1[3], oid_j=test_input_1[4],
                  mid_i=test_input_1[5], mid_j=test_input_1[6],
                  rid_i=test_input_1[7], rid_j=test_input_1[7],
                  eid_i=test_input_1[9], eid_j=test_input_1[10]) == expected_1
