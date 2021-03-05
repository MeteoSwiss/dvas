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
import pandas as pd

from dvas.tools.gdps.utils import corcoefs, weighted_mean, process_chunk
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME, PRF_REF_FLG_NAME
from dvas.hardcoded import PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME

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

def test_corcoefs(test_input_1, expected_1):
    """Function used to test if the GDP correlations are properly implemented.

    The function tests:
        - correlation coefficients for the different measurement types

    """

    # If I get here, then most likely it is all working fine.
    assert corcoefs(test_input_1[0], test_input_1[1], test_input_1[2],
                    oid_i=test_input_1[3], oid_j=test_input_1[4],
                    mid_i=test_input_1[5], mid_j=test_input_1[6],
                    rid_i=test_input_1[7], rid_j=test_input_1[7],
                    eid_i=test_input_1[9], eid_j=test_input_1[10]) == expected_1

@pytest.fixture
def chunk():
    """ A data chunk to test the GDP utils functions. """

    # First, the level 1 column names
    lvl_one = [PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME,
               PRF_REF_FLG_NAME, PRF_REF_UCR_NAME, PRF_REF_UCS_NAME,
               PRF_REF_UCT_NAME, PRF_REF_UCU_NAME, 'uc_tot', 'w_ps', 'oid', 'mid', 'eid', 'rid']

    # Set the proper MultiIndex
    cols = pd.MultiIndex.from_tuples([(ind, item) for item in lvl_one for ind in range(3)])

    # Initialize the DataFrame
    test_chunk = pd.DataFrame(index=pd.Series(range(10)),
                              columns=cols).sort_index(axis=1)

    # Set the proper types for the tdt columns
    test_chunk.loc[:, (slice(None), 'tdt')] = \
        test_chunk.loc[:, (slice(None), 'tdt')].astype('timedelta64[ns]')

    # Set the types for the other columns
    for key in lvl_one:
        if key in ['tdt', PRF_REF_FLG_NAME]:
            continue
        test_chunk.loc[:, (slice(None), key)] = \
            test_chunk.loc[:, (slice(None), key)].astype('float')

    # The time deltas
    test_chunk.loc[:, (0, PRF_REF_TDT_NAME)] = pd.to_timedelta(range(10), unit='s')
    test_chunk.loc[:, (1, PRF_REF_TDT_NAME)] = pd.to_timedelta(range(1, 11), unit='s')
    test_chunk.loc[:, (2, PRF_REF_TDT_NAME)] = pd.to_timedelta(np.arange(0.01, 10.01, 1), unit='s')

    # Some altitudes
    test_chunk.loc[:, (0, PRF_REF_ALT_NAME)] = np.arange(0, 50, 5.)
    test_chunk.loc[:, (1, PRF_REF_ALT_NAME)] = np.arange(1, 50, 5.)
    test_chunk.loc[:, (2, PRF_REF_ALT_NAME)] = np.arange(0.01, 50, 5.)

    # Some values
    test_chunk.loc[:, (0, PRF_REF_VAL_NAME)] = 1.
    test_chunk.loc[:, (1, PRF_REF_VAL_NAME)] = 2.
    test_chunk.loc[:, (2, PRF_REF_VAL_NAME)] = 4.
    test_chunk.loc[:, (slice(None), 'w_ps')] = 1.

    # Set some NaN's
    test_chunk.loc[1, (slice(None), PRF_REF_VAL_NAME)] = np.nan
    test_chunk.loc[2, (0, PRF_REF_VAL_NAME)] = np.nan
    test_chunk.loc[8, (0, 'w_ps')] = np.nan
    test_chunk.loc[9, (slice(None), 'w_ps')] = np.nan

    # Some errors
    test_chunk.loc[:, (slice(None), PRF_REF_UCR_NAME)] = 1.
    test_chunk.loc[:, (slice(None), PRF_REF_UCS_NAME)] = 1.
    test_chunk.loc[:, (slice(None), PRF_REF_UCT_NAME)] = 1.
    test_chunk.loc[:, (slice(None), PRF_REF_UCU_NAME)] = 1.
    test_chunk.loc[:, (slice(None), 'uc_tot')] = 2.

    # Errors are NaNs, but values are not.
    test_chunk.loc[9, (slice(None), PRF_REF_UCR_NAME)] = np.nan

    # THe other stuff
    test_chunk.loc[:, (slice(None), 'eid')] = 'e:1'
    test_chunk.loc[:, (slice(None), 'rid')] = 'r:1'

    for ind in range(3):
        test_chunk.loc[:, (ind, 'oid')] = ind
        test_chunk.loc[:, (ind, 'mid')] = ind

    return test_chunk

def test_weighted_mean(chunk):
    """ Function used to test the weighted_mean combination of profiles.

    """

    out, jac_mat = weighted_mean(chunk, binning=1)
    # Can I actually compute a weighted mean ?
    assert out.loc[0, PRF_REF_VAL_NAME] == 7/3
    assert out.loc[0, PRF_REF_TDT_NAME] == 1/3 * pd.to_timedelta(1.01, unit='s')
    assert out.loc[0, PRF_REF_ALT_NAME] == 1.01/3

    # If all the values are NaNs, should be NaN.
    assert np.isnan(out.loc[1, PRF_REF_VAL_NAME])
    # If only some of the bins are NaN's, I should return a number
    assert out.loc[2, PRF_REF_VAL_NAME] == 6/2
    # if all the weights are NaN's, return NaN
    assert np.isnan(out.loc[9, PRF_REF_VAL_NAME])
    # jac_mat has correct dimensions ?
    assert np.shape(jac_mat) == (int(np.ceil(len(chunk))), len(chunk)*3)
    # Content of jac_mat is as expected
    assert jac_mat[0, 0] == 1/3
    assert jac_mat[0, 10] == 1/3
    assert jac_mat[0, 20] == 1/3
    assert np.all(jac_mat[0, 1:10] == 0)
    assert np.all(jac_mat[0, 11:20] == 0)
    assert np.all(jac_mat[0, 21:] == 0)
    assert np.all(np.isnan(jac_mat[9, :]))

    # Idem but with some binning this time
    out, jac_mat = weighted_mean(chunk, binning=3)
    # Ignore the NaN values in the bin, and normalize properly
    assert out.loc[0, PRF_REF_VAL_NAME] == 13/5
    # Only valid values ... the easy stuff
    assert out.loc[1, PRF_REF_VAL_NAME] == 7/3
    # Do I actually have the correct amount of bins ?
    assert len(out) == 4
    # Is the last bin correct ?
    assert np.isnan(out.loc[3, PRF_REF_VAL_NAME])
    # Did I handle the Nan-weight ok ?
    assert out.loc[2, PRF_REF_VAL_NAME] == 20/8
    # jac_mat has correct dimensions ?
    assert np.shape(jac_mat) == (int(np.ceil(len(chunk)/3)), len(chunk)*3)
    assert jac_mat[0, 0] == 1/5


# TODO: fix the stuff below
#def test_delta():
#    """ Function used to test the weighted_mean combination of profiles.
#
#    """
#
#    vals = pd.DataFrame(np.ones((10, 2)))
#    vals[1] = 2.
#
#    # Can I actually compute a difference ?
#    assert np.all(tools.delta(vals, binning=1)[0] == -1)
#
#    vals.iloc[0] = np.nan
#    vals[0][9] = np.nan
#
#    # If all values for a bin a NaN, result should be NaN
#    assert np.isnan(tools.delta(vals, binning=1)[0][0])
#
#    # If only part of a bin is NaN, then report a number
#    assert tools.delta(vals, binning=2)[0][0] == -1
#
#    # If only on number is NaN, result should be Nan
#    assert np.isnan(tools.delta(vals, binning=1)[0][9])
#
#    # Check that the last bin is smaller than the others, I still compute it
#    assert len(tools.delta(vals, binning=4)[0]) == 3
#

def test_process_chunk(chunk):
    """ Function to test the processing of Profile chunks. This is the one responsible for the
    propagation of errors.
    """

    out = process_chunk(chunk, binning=1, method='mean')

    assert out.loc[0, PRF_REF_UCR_NAME] == np.sqrt(1/3)
    assert out.loc[0, PRF_REF_UCS_NAME] == 1
    assert out.loc[0, PRF_REF_UCT_NAME] == 1
    assert out.loc[0, PRF_REF_UCU_NAME] == np.sqrt(1/3)
    assert np.isnan(out.loc[1, PRF_REF_UCR_NAME]) # Values are all NaNs
    assert np.isnan(out.loc[9, PRF_REF_UCR_NAME]) # Error are all NaNs
