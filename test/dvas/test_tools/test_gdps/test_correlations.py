# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing 'gruan' classes and function of the tools submodule.

"""

# Import from python packages and modules
import numpy as np
import pytest

# Function to test
from dvas.tools.gdps.correlations import corr_coeff_matrix

# Define a series of reference measurement parameters, to test the different combination options.
T_OK = np.array([0, 0])                   # Same time
T_NOK = np.array([0, 1])                  # Different times
SRN_OK = np.array(['srn']*2)              # Same Serial Number
SRN_NOK = np.array(['srn a', 'srn b'])    # Different SRN
MDL_OK = np.array(['mod']*2)              # Same GDP model
MDL_NOK = np.array(['mod a', 'mod b'])    # Different GDP model
RIG_OK = np.array(['rig']*2)              # Same rig name
RIG_NOK = np.array(['rig a', 'rig b'])    # Different rig names
EVT_OK = np.array(['evt']*2)              # Same event
EVT_NOK = np.array(['evt x', 'evt y'])    # Different event


# Let us test a series of conditions for the different types of uncertainty types
@pytest.mark.parametrize("test_input_1, expected_1", [
    # Uncorrelated errors:
    (['ucu', T_OK, SRN_OK, MDL_OK, RIG_OK, EVT_OK], 1),
    # Uncorrelated errors: different time
    (['ucu', T_NOK, SRN_OK, MDL_OK, RIG_OK, EVT_OK], 0),
    # Uncorrelated errors: different Serial Number
    (['ucu', T_OK, SRN_NOK, MDL_OK, RIG_OK, EVT_OK], 0),
    # Uncorrelated errors: different rig
    (['ucu', T_OK, SRN_NOK, MDL_OK, RIG_NOK, EVT_OK], 0),
    # Uncorrelated errors: different events
    (['ucu', T_OK, SRN_NOK, MDL_OK, RIG_OK, EVT_NOK], 0),
    #
    # Spatial-correlated errors:
    (['ucs', T_OK, SRN_OK, MDL_OK, RIG_OK, EVT_OK], 1),
    # Spatial-correlated errors: different times
    (['ucs', T_NOK, SRN_NOK, MDL_OK, RIG_OK, EVT_OK], 1),
    # Uncorrelated errors: different rigs
    (['ucs', T_OK, SRN_NOK, MDL_OK, RIG_NOK, EVT_OK], 1),
    # Uncorrelated errors: different rigs  & different times
    (['ucs', T_NOK, SRN_NOK, MDL_OK, RIG_NOK, EVT_OK], 1),
    # Uncorrelated errors: different event
    (['ucs', T_OK, SRN_NOK, MDL_OK, RIG_OK, EVT_NOK], 0),
    #
    # Temporal-correlated errors:
    (['uct', T_OK, SRN_OK, MDL_OK, RIG_OK, EVT_OK], 1),
    # Temporal-correlated errors: different time
    (['uct', T_NOK, SRN_NOK, MDL_OK, RIG_OK, EVT_OK], 1),
    # Temporal-correlated errors: different rigs & different times
    (['uct', T_NOK, SRN_NOK, MDL_OK, RIG_NOK, EVT_OK], 1),
    # Temporal-correlated errors: different events and times
    (['uct', T_NOK, SRN_NOK, MDL_OK, RIG_OK, EVT_NOK], 1),
    ])
def test_corr_coeff_matrix(test_input_1, expected_1):
    """ Function used to test if the GDP correlations are properly implemented.

    The function tests:
        - correlation coefficients for the different measurement types

    """

    # If I get here, then most likely it is all working fine.
    out = corr_coeff_matrix(test_input_1[0], test_input_1[1], oids=test_input_1[2],
                            mids=test_input_1[3], rids=test_input_1[4], eids=test_input_1[5])

    assert out[0][1] == expected_1
    assert out[1][0] == expected_1
    assert np.array_equal(np.diag(out), np.ones(2))
