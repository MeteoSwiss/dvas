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
from dvas.hardcoded import PRF_TDT, PRF_ALT, PRF_VAL, PRF_FLG, PRF_UCR, PRF_UCS, PRF_UCT, PRF_UCU

# Function to test
from dvas.tools.gdps.utils import weighted_mean, delta, process_chunk


@pytest.fixture
def chunk():
    """ A data chunk to test the GDP utils functions. """

    # First, the level 1 column names
    lvl_one = [PRF_TDT, PRF_ALT, PRF_VAL, PRF_FLG, PRF_UCR, PRF_UCS, PRF_UCT, PRF_UCU,
               'uc_tot', 'w_ps', 'oid', 'mid', 'eid', 'rid']

    # Set the proper MultiIndex
    cols = pd.MultiIndex.from_tuples([(ind, item) for item in lvl_one for ind in range(3)])

    # Initialize the DataFrame
    test_chunk = pd.DataFrame(index=pd.Series(range(10)), columns=cols).sort_index(axis=1)

    # Set the proper types for the different columns
    for key in lvl_one:
        if key == 'tdt':
            test_chunk.loc[:, (slice(None), key)] = \
                test_chunk.loc[:, (slice(None), key)].astype('timedelta64[ns]')
        elif key == PRF_FLG:
            test_chunk.loc[:, (slice(None), key)] = 0
            test_chunk.loc[:, (slice(None), key)] = \
                test_chunk.loc[:, (slice(None), key)].astype(int)
        else:
            test_chunk.loc[:, (slice(None), key)] = \
                test_chunk.loc[:, (slice(None), key)].astype('float')

    # The time deltas
    test_chunk.loc[:, (0, PRF_TDT)] = pd.to_timedelta(range(10), unit='s')
    test_chunk.loc[:, (1, PRF_TDT)] = pd.to_timedelta(range(1, 11), unit='s')
    test_chunk.loc[:, (2, PRF_TDT)] = pd.to_timedelta(np.arange(0.01, 10.01, 1), unit='s')

    # Some altitudes
    test_chunk.loc[:, (0, PRF_ALT)] = np.arange(0, 50, 5.)
    test_chunk.loc[:, (1, PRF_ALT)] = np.arange(1, 50, 5.)
    test_chunk.loc[:, (2, PRF_ALT)] = np.arange(0.01, 50, 5.)

    # Some values
    test_chunk.loc[:, (0, PRF_VAL)] = 1.
    test_chunk.loc[:, (1, PRF_VAL)] = 2.
    test_chunk.loc[:, (2, PRF_VAL)] = 4.
    test_chunk.loc[:, (slice(None), 'w_ps')] = 1.

    # Set some NaN's
    test_chunk.loc[1, (slice(None), PRF_VAL)] = np.nan
    test_chunk.loc[8:9, (0, PRF_VAL)] = np.nan
    test_chunk.loc[2, (0, PRF_VAL)] = np.nan
    test_chunk.loc[8, (0, 'w_ps')] = np.nan
    test_chunk.loc[9, (slice(None), 'w_ps')] = np.nan

    # Some errors
    test_chunk.loc[:, (slice(None), PRF_UCR)] = 1.
    test_chunk.loc[:, (slice(None), PRF_UCS)] = 1.
    test_chunk.loc[:, (slice(None), PRF_UCT)] = 1.
    test_chunk.loc[:, (slice(None), PRF_UCU)] = 1.
    test_chunk.loc[:, (slice(None), 'uc_tot')] = 2.

    # Some flags
    test_chunk.loc[0, (0, PRF_FLG)] = 1
    test_chunk.loc[0, (1, PRF_FLG)] = 2
    test_chunk.loc[0, (2, PRF_FLG)] = 4
    test_chunk.loc[1, (1, PRF_FLG)] = 8
    test_chunk.loc[2, (1, PRF_FLG)] = 0
    test_chunk.loc[2, (2, PRF_FLG)] = 3
    test_chunk.loc[9, (0, PRF_FLG)] = 0
    test_chunk.loc[8, (0, PRF_FLG)] = 1
    test_chunk.loc[8, (1, PRF_FLG)] = 2
    test_chunk.loc[8, (2, PRF_FLG)] = 4

    # Errors are NaNs, but values are not.
    test_chunk.loc[9, (slice(None), PRF_UCR)] = np.nan

    # THe other stuff
    test_chunk.loc[:, (slice(None), 'eid')] = 'e:1'
    test_chunk.loc[:, (slice(None), 'rid')] = 'r:1'

    for ind in range(3):
        test_chunk.loc[:, (ind, 'oid')] = ind
        test_chunk.loc[:, (ind, 'mid')] = 'A'  # Force the same mid for all Profiles

    return test_chunk


def test_weighted_mean(chunk):
    """ Function used to test the weighted_mean combination of profiles."""

    out, jac_mat = weighted_mean(chunk, binning=1)
    # Can I actually compute a weighted mean ?
    assert out.loc[0, PRF_VAL] == 7/3
    assert out.loc[0, PRF_TDT] == 1/3 * pd.to_timedelta(1.01, unit='s')
    assert out.loc[0, PRF_ALT] == 1.01/3

    # If all the values are NaNs, should be NaN, and so should the flag.
    assert out.isna().loc[1, PRF_VAL]
    assert out.loc[1, PRF_FLG] == 0
    # If only some of the bins are NaN's, I should return a number and flag
    assert out.loc[2, PRF_VAL] == 6/2
    assert out.loc[2, PRF_FLG] == 3
    # if all the weights are NaN's, return NaN
    assert out.isna().loc[9, PRF_VAL]
    assert out.loc[9, PRF_FLG] == 0
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
    # Ignore the NaN values in the bin, and normalize properly. Combine flags just fine.
    assert out.loc[0, PRF_VAL] == 13/5
    assert out.loc[0, PRF_FLG] == 7
    # Only valid values ... the easy stuff
    assert out.loc[1, PRF_VAL] == 7/3
    # Do I actually have the correct amount of bins ?
    assert len(out) == 4
    # Is the last bin correct ?
    assert np.isnan(out.loc[3, PRF_VAL])
    # Did I handle the Nan-weight ok ?
    assert out.loc[2, PRF_VAL] == 20/8
    # And the partial flags ?
    assert out.loc[2, PRF_FLG] == 6
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
    assert out.loc[0, 'val'] == 1  # Can I compute a delta ?
    assert np.isnan(out.loc[1, 'val'])  # What if I have two NaNs ?
    assert np.isnan(out.loc[2, 'val'])  # What if I have partial NaN ?
    # Correct jacobian shape ?
    assert np.shape(jac_out) == (len(chunk_2)//binning + len(chunk_2) % binning, 2*len(chunk_2))
    # Correct flags ?
    assert out.loc[0, 'flg'] == 3
    assert out.loc[1, 'flg'] == 0
    assert out.loc[2, 'flg'] == 0
    assert out.loc[9, 'flg'] == 0

    # Now do the same with some binning
    binning = 2
    out, jac_out = delta(chunk_2, binning=binning)
    # Correct Jacobian shape ?
    assert np.shape(jac_out) == (len(chunk_2)//binning + len(chunk_2) % binning, 2*len(chunk_2))
    # Partial NaN's are ignored ?
    assert out.loc[0, 'val'] == 1
    # Full NaN's are handled properly ?
    assert np.isnan(out.loc[4, 'val'])
    # Correct flags
    assert out.loc[0, 'flg'] == 3
    assert out.loc[1, 'flg'] == 0
    assert all(out.isna()[2:])


def test_process_chunk(chunk):
    """ Function to test the processing of Profile chunks. This is the one responsible for the
    propagation of errors.
    """

    # First test the mean
    out_1, _ = process_chunk(chunk, binning=1, method='weighted mean')
    assert out_1.loc[0, PRF_UCR] == np.sqrt(1/3)
    assert out_1.loc[0, PRF_UCS] == 1
    assert out_1.loc[0, PRF_UCT] == 1
    assert out_1.loc[0, PRF_UCU] == np.sqrt(1/3)
    assert np.isnan(out_1.loc[1, PRF_UCR])  # Values are all NaNs
    assert np.isnan(out_1.loc[9, PRF_UCR])  # Error are all NaNs

    # With partial NaN's, errors still get computed correctly.
    assert not np.isnan(out_1.loc[8, PRF_UCR])
    assert not np.isnan(out_1.loc[8, PRF_VAL])

    # Now with binning
    out_2, _ = process_chunk(chunk, binning=2, method='mean')
    # Correct length ?
    assert len(out_2) == len(out_1)//2 + len(out_1) % 2

    # Then assess the delta
    out_1, _ = process_chunk(chunk.loc[:, :1], binning=1, method='delta')

    # If some crazy users has NaN's for values but non-NaN errors, make sure I fully ignore these.
    assert np.isnan(out_1.loc[1, 'ucr'])

    # What happens with binning ?
    out_2, _ = process_chunk(chunk.loc[:, :1], binning=2, method='delta')

    # In case of partial binning things get ignored accordingly.
    assert all(out_1.iloc[0].values == out_2.iloc[0].values)
    # If all is NaN, then result is NaN.
    assert all(out_2.loc[4, ['val', 'ucr', 'ucs', 'uct', 'ucu']].isna())
