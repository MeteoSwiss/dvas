# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function of the gruan submodule.

"""

# Import from python packages and modules
import numpy as np
import pytest
from dvas.gruan import corcoef_gdp

# Define a series of reference measurement parameters, to test the different combination options.
T_OK = [np.zeros(1), np.zeros(1)]                        # Same time
T_NOK = [np.ones(1), np.zeros(1)]                        # Different times
SN_OK = [np.array(['sn']), np.array(['sn'])]             # Same Serial Number
SN_NOK = [np.array(['sn']), np.array(['sns'])]           # Different SN
RIG_OK = [np.array(['rig a']), np.array(['rig a'])]      # Same rig name
RIG_NOK = [np.array(['rig a']), np.array(['rig b'])]     # Different rig names
EVT_OK = [np.array(['event x']), np.array(['event x'])]  # Same event
EVT_NOK = [np.array(['event x']), np.array(['event y'])] # Different event
SIT_OK = [np.array(['site 1']), np.array(['site 1'])]    # Same site
SIT_NOK = [np.array(['site 1']), np.array(['site 2'])]   # Different site


# Let us test a series of conditions for the different types of uncertainty types
@pytest.mark.parametrize("test_input, expected",
                         [   # Uncorrelated errors: same point
                             (T_OK + ['sigma_u'] + SN_OK + RIG_OK + EVT_OK + SIT_OK, 1),
                             # Uncorrelated errors: different time
                             (T_NOK + ['sigma_u'] + SN_OK + RIG_OK + EVT_OK + SIT_OK, 0),
                             # Uncorrelated errors: different Serial Number
                             (T_OK + ['sigma_u'] + SN_NOK + RIG_OK + EVT_OK + SIT_OK, 0),
                             # Uncorrelated errors: different rig
                             (T_OK + ['sigma_u'] + SN_OK + RIG_NOK + EVT_OK + SIT_OK, 0),
                             # Uncorrelated errors: different event
                             (T_OK + ['sigma_u'] + SN_OK + RIG_OK + EVT_NOK + SIT_OK, 0),
                             # Uncorrelated errors: different site
                             (T_OK + ['sigma_u'] + SN_OK + RIG_OK + EVT_OK + SIT_NOK, 0),
                             #
                             # Environmental-correlated errors:
                             # TODO
                             #
                             # Spatial-correlated errors: same point
                             (T_OK + ['sigma_s'] + SN_OK + RIG_OK + EVT_OK + SIT_OK, 1),
                             # Spatial-correlated errors: different times
                             (T_NOK + ['sigma_s'] + SN_NOK + RIG_OK + EVT_OK + SIT_OK, 1),
                             # Uncorrelated errors: different rigs
                             (T_OK + ['sigma_s'] + SN_NOK + RIG_NOK + EVT_OK + SIT_OK, 1),
                             # Uncorrelated errors: different rigs  & different times
                             (T_NOK + ['sigma_s'] + SN_NOK + RIG_NOK + EVT_OK + SIT_OK, 1),
                             # Uncorrelated errors: different event
                             (T_OK + ['sigma_s'] + SN_NOK + RIG_OK + EVT_NOK + SIT_OK, 0),
                             # Uncorrelated errors: different event
                             (T_OK + ['sigma_s'] + SN_NOK + RIG_OK + EVT_OK + SIT_NOK, 0),
                             #
                             # Temporal-correlated errors: same point
                             (T_OK + ['sigma_t'] + SN_OK + RIG_OK + EVT_OK + SIT_OK, 1),
                             # Temporal-correlated errors: different time
                             (T_NOK + ['sigma_t'] + SN_NOK + RIG_OK + EVT_OK + SIT_OK, 1),
                             # Temporal-correlated errors: different rigs & different times
                             (T_NOK + ['sigma_t'] + SN_NOK + RIG_NOK + EVT_OK + SIT_OK, 1),
                             # Temporal-correlated errors: different events and times
                             (T_NOK + ['sigma_t'] + SN_NOK + RIG_OK + EVT_NOK + SIT_OK, 1),
                             # Temporal-correlated errors: different sites and times
                             (T_NOK + ['sigma_t'] + SN_NOK + RIG_OK + EVT_OK + SIT_NOK, 1),
                         ])

def test_gruan_corcoef_gdp(test_input, expected):
    """Function used to test if the GDP correlations are properly implemented.

    The function tests:
        - correlation coefficients for the different measurement types

    """

    # If I get here, then most likely it is all working fine.
    assert corcoef_gdp(test_input[0], test_input[1], test_input[2],
                       sn_i=test_input[3], sn_j=test_input[4],
                       rig_i=test_input[5], rig_j=test_input[6],
                       event_i=test_input[7], event_j=test_input[8],
                       site_i=test_input[9], site_j=test_input[10]) == expected
