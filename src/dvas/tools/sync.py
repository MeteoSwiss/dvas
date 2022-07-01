# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains tools to synchronize profiles.

"""

# Import from other Python packages
import logging
import warnings
import numpy as np

# Import from this package
from ..logger import log_func_call
from ..errors import DvasError
from ..hardcoded import PRF_ALT, PRF_VAL, PRF_TDT, MTDTA_START

# Setup the local logger
logger = logging.getLogger(__name__)


@log_func_call(logger)
def get_sync_shifts_from_starttime(prfs):
    """ A routine that estimates the necessary synchronization shifts between profiles based on the
    GPS start time value.

    Args:
        prfs (dvas.data.data.MultiRSProfiles): list of Profiles to sync.

    Returns:
        list of int: list of shifts required to sync the profiles with each other.
    """

    # Start with some sanity checks
    for prf in prfs:
        if MTDTA_START not in prf.info.metadata.keys():
            raise DvasError(f"'{MTDTA_START}' not found in metadata for: {prf.info.src}")
        if prf.info.metadata[MTDTA_START] is None:
            raise DvasError(
                f"'{MTDTA_START}' is None. GPS start-time sync impossible for: {prf.info.src}")

    # Extract all the start times, and convert them to datetimes
    start_times = [prf.info.metadata[MTDTA_START] for prf in prfs]

    shifts = [int(np.round((item-np.min(start_times)).total_seconds())) for item in start_times]

    return shifts


@log_func_call(logger)
def get_sync_shifts_from_alt(prfs, ref_alt=5000.):
    """ A routine that estimates the shifts required to synchronize profiles, based on the
    the altitude index.

    This is a very crude function that does the sync based on a single altitude, and thus happily
    ignore any drift/stretch of any kind.

    Args:
        prfs (dvas.data.data.MultiRSProfiles): list of Profiles to compare
        ref_alt (float): the altitude at which to sync the profiles.

    Returns:
        list of int: list of shifts required to synchronize profiles in order to match ref_alt
            Sign convention: profiles are synchronized when row n goes to row n+shift

    """

    # First, find the measured altitude that is closest to the reference altitude for the first
    # profile. Set this as the reference altitude, to make sure we match a real datapoint.

    # Extract the altitudes
    alts = prfs.get_prms(PRF_ALT)

    # What is the index of the first profile that best matches the ref_alt ?
    ref_ind_0 = (alts[0]-ref_alt).abs().idxmin()

    # Update the ref_alt accordingly: this is to ensure that we match profiles between each others.
    # And not to an external value, that could lead to a different alignment in specific cases.
    ref_alt_mod = float(alts.loc[ref_ind_0, 0].values)

    # Now, find the corresponding index for all the profiles
    out = (alts-ref_alt_mod).abs().idxmin(axis=0).values

    # Convert this to shifts such that profiles are synced if row n goes to row n+shift
    out = np.max(out) - out

    # Make sure to keep them all positive
    out -= np.min(out)
    # Return the corresponding shifts, not forgetting that the first profile stays where it is.
    return list(out)


@log_func_call(logger)
def get_sync_shifts_from_val(prfs, max_shift=100, first_guess=None):
    """ Estimates the shifts required to synchronize profiles, such that <abs(val_A-val_B)> is
    minimized.

    Args:
        prfs (dvas.data.data.MultiRSProfiles): list of Profiles to compare
        max_shift (int, optional): maximum (absolute) shift to consider. Must be positive.
            Defaults to 100.
        first_guess (int|list of int, optional): starting guess around which to center the search.
            Defaults to None.

    Returns:
        list of int: list of shifts required to synchronize profiles.
            Sign convention: profiles are synchronized when row n goes to row n+shift

    """

    # Start with some sanity checks
    if first_guess is None:
        first_guess = 0
    if isinstance(first_guess, int):
        first_guess = [first_guess] * len(prfs)
    if len(first_guess) != len(prfs):
        raise DvasError('Ouch ! first guess should be the same length as prfs.')

    # In what follows, we assume that all the profiles are sampled with a fixed, uniform timestep.
    # Let's raise an error if this is not the case.
    # To do that, let's compute the differences between the different time steps, and check if it
    # is unique (or not) and identical for all profiles (or Not)
    ndts = prfs.get_prms(PRF_TDT).diff(periods=1, axis=0).nunique(axis=0, dropna=True)
    if np.any([item != 1 for item in ndts.values]):
        raise DvasError(f'Ouch ! The profiles do not all have uniform time steps: {ndts.values}')
    dts = [prfs.get_prms(PRF_TDT)[item][PRF_TDT].diff().unique().tolist()
           for item in range(len(prfs))]
    if np.any([item != dts[0] for item in dts]):
        raise DvasError(f'Ouch ! Inconsistent time steps between the different profiles: {dts}')

    # Let us first begin by extracting all the value arrays that need to be "cross-correlated".
    vals = prfs.get_prms(PRF_VAL)

    # Let's build an array of shifts to consider
    shifts = np.arange(-np.abs(max_shift), np.abs(max_shift)+1, 1)

    # Now, let's find the shifts that give rise to the smallest mean absolute offsets between the
    # profiles. If the shifts are larger than the array, I get a RunTime warning from numpy.
    # Let's just hide these away shamefully.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        ind = [[np.nanmean(np.abs(1.-vals[0]/vals.shift(shift)[ind]))
                for ind in range(len(prfs))] for shift in shifts]

    ind = np.nanargmin(np.array(ind), axis=0)

    # Issue some warning if I am at the very edge of the search range
    if np.any(max_shift-np.abs(shifts[ind]) < 10):
        logger.warning('sync_shift_from_val values is close from the edge of the search zone')

    # Return a list of shifts, resetting it to only have positive shifts.
    out = list(shifts[ind]-np.min(shifts[ind]))

    # Check if we are far from the first_guess ... and if so, raise a warning
    if any(np.abs(item - first_guess[i]) > 3 for i, item in enumerate(out)):
        logger.warning('Large offset from first guess: %s vs %s', out, first_guess)

    return out
