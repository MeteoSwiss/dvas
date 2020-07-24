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
T_OK = [np.zeros(1), np.zeros(1)]                       # Same time
T_NOK = [np.ones(1), np.zeros(1)]                       # Different times
SRN_OK = [np.array(['srn']), np.array(['srn'])]         # Same Serial Number
SRN_NOK = [np.array(['srn a']), np.array(['srn b'])]    # Different SRN
MOD_OK = [np.array(['mod']), np.array(['mod'])]         # Same GDP model
MOD_NOK = [np.array(['mod a']), np.array(['mod b'])]    # Different GDP model
RIG_OK = [np.array(['rig']), np.array(['rig'])]         # Same rig name
RIG_NOK = [np.array(['rig a']), np.array(['rig b'])]    # Different rig names
EVT_OK = [np.array(['evt']), np.array(['evt'])]         # Same event
EVT_NOK = [np.array(['evt x']), np.array(['evt y'])]    # Different event
SIT_OK = [np.array(['sit']), np.array(['sit'])]         # Same site
SIT_NOK = [np.array(['sit 1']), np.array(['sit 2'])]    # Different site

# Let us test a series of conditions for the different types of uncertainty types
@pytest.mark.parametrize("test_input_1, expected_1",
                         [   # Uncorrelated errors: same point
                             (T_OK + ['sigma_u'] + SRN_OK + MOD_OK + RIG_OK + EVT_OK + SIT_OK, 1),
                             # Uncorrelated errors: different time
                             (T_NOK + ['sigma_u'] + SRN_OK + MOD_OK + RIG_OK + EVT_OK + SIT_OK, 0),
                             # Uncorrelated errors: different Serial Number
                             (T_OK + ['sigma_u'] + SRN_NOK + MOD_OK + RIG_OK + EVT_OK + SIT_OK, 0),
                             # Uncorrelated errors: different rig
                             (T_OK + ['sigma_u'] + SRN_NOK + MOD_OK + RIG_NOK + EVT_OK + SIT_OK, 0),
                             # Uncorrelated errors: different event
                             (T_OK + ['sigma_u'] + SRN_NOK + MOD_OK + RIG_OK + EVT_NOK + SIT_OK, 0),
                             # Uncorrelated errors: different site
                             (T_OK + ['sigma_u'] + SRN_NOK + MOD_OK + RIG_OK + EVT_OK + SIT_NOK, 0),
                             #
                             # Environmental-correlated errors:
                             # TODO
                             #
                             # Spatial-correlated errors: same point
                             (T_OK + ['sigma_s'] + SRN_OK + MOD_OK + RIG_OK + EVT_OK + SIT_OK, 1),
                             # Spatial-correlated errors: different times
                             (T_NOK + ['sigma_s'] + SRN_NOK + MOD_OK + RIG_OK + EVT_OK + SIT_OK, 1),
                             # Uncorrelated errors: different rigs
                             (T_OK + ['sigma_s'] + SRN_NOK + MOD_OK + RIG_NOK + EVT_OK + SIT_OK, 1),
                             # Uncorrelated errors: different rigs  & different times
                             (T_NOK + ['sigma_s'] + SRN_NOK + MOD_OK + RIG_NOK + EVT_OK + SIT_OK, 1),
                             # Uncorrelated errors: different event
                             (T_OK + ['sigma_s'] + SRN_NOK + MOD_OK + RIG_OK + EVT_NOK + SIT_OK, 0),
                             # Uncorrelated errors: different event
                             (T_OK + ['sigma_s'] + SRN_NOK + MOD_OK + RIG_OK + EVT_OK + SIT_NOK, 0),
                             #
                             # Temporal-correlated errors: same point
                             (T_OK + ['sigma_t'] + SRN_OK + MOD_OK + RIG_OK + EVT_OK + SIT_OK, 1),
                             # Temporal-correlated errors: different time
                             (T_NOK + ['sigma_t'] + SRN_NOK + MOD_OK + RIG_OK + EVT_OK + SIT_OK, 1),
                             # Temporal-correlated errors: different rigs & different times
                             (T_NOK + ['sigma_t'] + SRN_NOK + MOD_OK + RIG_NOK + EVT_OK + SIT_OK, 1),
                             # Temporal-correlated errors: different events and times
                             (T_NOK + ['sigma_t'] + SRN_NOK + MOD_OK + RIG_OK + EVT_NOK + SIT_OK, 1),
                             # Temporal-correlated errors: different sites and times
                             (T_NOK + ['sigma_t'] + SRN_NOK + MOD_OK + RIG_OK + EVT_OK + SIT_NOK, 1),
                         ])

def test_gruan_corcoef_gdp(test_input_1, expected_1):
    """Function used to test if the GDP correlations are properly implemented.

    The function tests:
        - correlation coefficients for the different measurement types

    """

    # If I get here, then most likely it is all working fine.
    assert corcoef_gdps(test_input_1[0], test_input_1[1], test_input_1[2],
                       srn_i=test_input_1[3], srn_j=test_input_1[4],
                       mod_i=test_input_1[5], mod_j=test_input_1[6],
                       rig_i=test_input_1[7], rig_j=test_input_1[7],
                       evt_i=test_input_1[9], evt_j=test_input_1[10],
                       sit_i=test_input_1[11], sit_j=test_input_1[12]) == expected_1


# Let us test a series of conditions for the different types of uncertainty types
@pytest.mark.parametrize("test_input_2, expected_2",
                         [   # Weighted mean binning of a single profile
                             ([np.arange(1, 3, 1), np.arange(1, 3, 1), np.arange(1, 3, 1), np.arange(1, 3, 1), np.arange(1, 3, 1), 2],
                              (np.array([1.2]), np.sqrt(112/25), [[0, 1]], np.array([0.5]))),
                             # Idem, but for a longer profile to validate the 

                         ])


def test_gruan_rebin_gdps(test_input_2, expected_2):
    """Function used to test if the routine combining GDP profiles is ok.

    The function tests:
        - correct propagation of errors when rebining a single profile

    Note:
       Inspired, in part, from `this post 
       <https://stackoverflow.com/questions/39896716/can-i-perform-multiple-assertions-in-pytest>`__
       by Jon Clements.

    """
    errors = []

    out = merge_andor_rebin_gdps([test_input_2[0]], # profiles
                                 [test_input_2[1]],  #sigma_us
                                 [test_input_2[2]], #sigma_es
                                 [test_input_2[3]], #sigma_ss
                                 [test_input_2[4]], #sigma_ts
                                 srns=None,
                                 mods=None,
                                 rigs=None,
                                 evts=None,
                                 sits=None,
                                 binning=test_input_2[5],
                                 method='weighted mean')

    if np.all(out[0] != expected_2[0]):
        errors += ['Merged profile is wrong']
    if np.all(np.sqrt(out[1]**2 + out[2]**2 + out[3]**2 + out[4]**2) != expected_2[1]):
        errors += ['sigma_tot is wrong']
    if np.all(out[5] != expected_2[2]):
        errors += ['old_inds is wrong']
    if np.all(out[6] != expected_2[3]):
        errors += ['new_ind is wrong']
    
    assert not errors


# Let us test a series of conditions for the different types of uncertainty types
@pytest.mark.parametrize("test_input_3, expected_3",
                         [   # Delta from two profiles from the same rig/event/site
                             # Sigma_s and sigma_t cancel perfectly. 
                             (SRN_NOK + MOD_OK + RIG_OK + EVT_OK + SIT_OK + [1, 'delta'],
                              (np.ones(5), np.sqrt(2)*np.ones(5), None, np.zeros(5), np.zeros(5),
                              [np.array([i]) for i in range(5)], range(5))),
                             # Delta + binning from two profiles from the same rig/event/site
                             (SRN_NOK + MOD_OK + RIG_OK + EVT_OK + SIT_OK + [3, 'delta'],
                              (np.ones(2), np.array([np.sqrt(6)/3, np.sqrt(4)/2]), None,
                               np.zeros(2), np.zeros(2), [[0, 1, 2], [3, 4]], np.array([1, 3.5]))),

                         ])

def test_gruan_merge_andor_rebin_gdps(test_input_3, expected_3):
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
                                 srns=test_input_3[:2],
                                 mods=test_input_3[2:4],
                                 rigs=test_input_3[4:6],
                                 evts=test_input_3[6:8],
                                 sits=test_input_3[8:10],
                                 binning=test_input_3[10],
                                 method=test_input_3[11])

    if np.all(out[0] != expected_3[0]):
        errors += ['Merged profile is wrong']
    if np.all(out[1] != expected_3[1]):
        errors += ['sigma_us_new is wrong']
    # TODO: sigma_es
    if np.all(out[3] != expected_3[3]):
        errors += ['sigma_ss_new is wrong']
    if np.all(out[4] != expected_3[4]):
        errors += ['sigma_ts_new is wrong']
    if np.all(out[5] != expected_3[5]):
        errors += ['old_inds is wrong']
    if np.all(out[6] != expected_3[6]):
        errors += ['new_ind is wrong']

    assert not errors
