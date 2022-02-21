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
import pandas as pd

from dvas.errors import DvasError
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME, PRF_REF_FLG_NAME
from dvas.hardcoded import PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME

# Function to test
from dvas.tools.gdps.utils import weighted_mean, delta, process_chunk

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
    test_chunk.loc[8:9, (0, PRF_REF_VAL_NAME)] = np.nan
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
    """ Function used to test the weighted_mean combination of profiles."""

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
    assert all(jac_mat[0, 1:10].mask)
    assert all(jac_mat[0, 11:20].mask)
    assert all(jac_mat[0, 21:].mask)
    assert all(jac_mat[9, :].mask)

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

def test_delta(chunk):
    """ Function used to test the delta combination of profiles."""

    # Check that things fail cleanly if more than 1 profile is being fed.
    with pytest.raises(DvasError):
        out = delta(chunk, binning=1)

    # No fancy binning
    chunk_2 = chunk.loc[:, :1]
    binning = 1
    out, jac_out = delta(chunk_2, binning=binning)
    assert len(out) == len(chunk)
    assert out.loc[0, 'val'] == 1 # Can I compute a delta ?
    assert np.isnan(out.loc[1, 'val']) # What if I have a partial NaN ?
    assert np.isnan(out.loc[2, 'val']) # What if I have two NaNs ?
    # Correct jacobian shape ?
    assert np.shape(jac_out) == (len(chunk_2)//binning + len(chunk_2)%binning, 2*len(chunk_2))

    # Now do the same with some binning
    binning = 2
    out, jac_out = delta(chunk_2, binning=binning)
    # Correct Jacobian shape ?
    assert np.shape(jac_out) == (len(chunk_2)//binning + len(chunk_2)%binning, 2*len(chunk_2))
    # Partial NaN's are ignored ?
    assert out.loc[0, 'val'] == 1
    # Full NaN's are handled properly ?
    assert np.isnan(out.loc[4, 'val'])

def test_process_chunk(chunk):
    """ Function to test the processing of Profile chunks. This is the one responsible for the
    propagation of errors.
    """

    # First test the mean
    out_1 = process_chunk(chunk, binning=1, method='weighted mean')
    assert out_1.loc[0, PRF_REF_UCR_NAME] == np.sqrt(1/3)
    assert out_1.loc[0, PRF_REF_UCS_NAME] == 1
    assert out_1.loc[0, PRF_REF_UCT_NAME] == 1
    assert out_1.loc[0, PRF_REF_UCU_NAME] == np.sqrt(1/3)
    assert np.isnan(out_1.loc[1, PRF_REF_UCR_NAME]) # Values are all NaNs
    assert np.isnan(out_1.loc[9, PRF_REF_UCR_NAME]) # Error are all NaNs

    ## With partial NaN's, errors still get computed correctly.
    assert not(np.isnan(out_1.loc[8, PRF_REF_UCR_NAME]))
    assert not(np.isnan(out_1.loc[8, PRF_REF_VAL_NAME]))

    # Now with binning
    out_2 = process_chunk(chunk, binning=2, method='mean')
    # Correct length ?
    assert len(out_2) == len(out_1)//2 + len(out_1) % 2

    # Then assess the delta
    out_1 = process_chunk(chunk.loc[:, :1], binning=1, method='delta')

    # If some crazy users has NaN's for values but non-NaN errors, make sure I fully ignore these.
    assert np.isnan(out_1.loc[1, 'ucr'])

    # What happens with binning ?
    out_2 = process_chunk(chunk.loc[:, :1], binning=2, method='delta')

    # In case of partial binning things get ignored accordingly.
    assert all(out_1.iloc[0].values == out_2.iloc[0].values)
    # If all is NaN, then result is NaN.
    assert all(out_2.loc[4, ['val', 'ucr', 'ucs', 'uct', 'ucu']].isna())
