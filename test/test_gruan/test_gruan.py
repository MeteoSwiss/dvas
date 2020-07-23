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
from dvas.gruan import corcoef_gdps, merge_andor_rebin_gdps

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
@pytest.mark.parametrize("test_input_1, expected_1",
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

def test_gruan_corcoef_gdp(test_input_1, expected_1):
    """Function used to test if the GDP correlations are properly implemented.

    The function tests:
        - correlation coefficients for the different measurement types

    """

    # If I get here, then most likely it is all working fine.
    assert corcoef_gdps(test_input_1[0], test_input_1[1], test_input_1[2],
                       sn_i=test_input_1[3], sn_j=test_input_1[4],
                       rig_i=test_input_1[5], rig_j=test_input_1[6],
                       event_i=test_input_1[7], event_j=test_input_1[8],
                       site_i=test_input_1[9], site_j=test_input_1[10]) == expected_1

# Let us test a series of conditions for the different types of uncertainty types
@pytest.mark.parametrize("test_input_2, expected_2",
                         [   # Delta from two profiles from the same rig/event/site
                             # Sigma_s and sigma_t cancel perfectly. 
                             (SN_NOK + RIG_OK + EVT_OK + SIT_OK + [1, 'delta'],
                              (np.ones(5), np.sqrt(2)*np.ones(5), None, np.zeros(5), np.zeros(5),
                              [np.array([i]) for i in range(5)], range(5))),
                         ])

def test_gruan_merge_andor_rebin_gdps(test_input_2, expected_2):
    """Function used to test if the routine combining GDP profiles is ok.

    The function tests:
        - correct propagation of errors

    Note:
       Inspired, in part, from `this post 
       <https://stackoverflow.com/questions/39896716/can-i-perform-multiple-assertions-in-pytest>`__
       by Jon Clements.

    Todo:
        * expand lists of tests to all uncertainty types.

    """
    errors = []

    out = merge_andor_rebin_gdps([np.ones(5), np.zeros(5)], # profiles
                                 [np.ones(5), np.ones(5)],  #sigma_us
                                 [np.ones(5), np.ones(5)], #sigma_es
                                 [np.ones(5), np.ones(5)], #sigma_ss
                                 [np.ones(5), np.ones(5)], #sigma_ts
                                 sns=test_input_2[:2],
                                 rigs=test_input_2[2:4],
                                 evts=test_input_2[4:6],
                                 sites=test_input_2[6:8],
                                 binning=test_input_2[8],
                                 method=test_input_2[9])

    if np.all(out[0] != expected_2[0]):
        errors += ['Merged profile is wrong']
    if np.all(out[1] != expected_2[1]):
        errors += ['sigma_us_new is wrong']
    # TODO: sigma_es
    if np.all(out[3] != expected_2[3]):
        errors += ['sigma_ss_new is wrong']
    if np.all(out[4] != expected_2[4]):
        errors += ['sigma_ts_new is wrong']
    if np.all(out[5] != expected_2[5]):
        errors += ['old_inds is wrong']
    if np.all(out[6] != expected_2[6]):
        errors += ['new_ind is wrong']

    import pdb
    pdb.set_trace()

    assert not errors
                                  
