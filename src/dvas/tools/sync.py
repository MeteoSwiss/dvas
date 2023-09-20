# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

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
from ..hardcoded import PRF_ALT, PRF_VAL, PRF_TDT, MTDTA_FIRST

# Setup the local logger
logger = logging.getLogger(__name__)


@log_func_call(logger)
def get_sync_shifts_from_time(prfs):
    """ A routine that estimates the necessary synchronization shifts between profiles based on the
    measurement times.

    Args:
        prfs (dvas.data.data.MultiRSProfiles): list of Profiles to sync.

    Returns:
        list of int: list of shifts required to sync the profiles with each other.
    """

    # Start with some sanity checks
    for prf in prfs:
        if MTDTA_FIRST not in prf.info.metadata.keys():
            raise DvasError(f"'{MTDTA_FIRST}' not found in metadata for: {prf.info.src}")

    # Extract all the measurement times of the first profile points, not forgetting to account for
    # any data cropping that may have taken place since it was loaded (i.e. get the MTDTA_FIRST
    # from the metadata, and add the first time delta - which should be 0 unless descent data was
    # cropped).
    start_times = [None if prf.info.metadata[MTDTA_FIRST] is None else
                   prf.info.metadata[MTDTA_FIRST] + prf.data.index.get_level_values(PRF_TDT)[0]
                   for prf in prfs]

    # Compute the corresponding shifts, resetting them to be all positive
    valid_start_time = [item for item in start_times if item is not None]
    shifts = [np.nan if (item is None) else
              int(np.round((item - np.min(valid_start_time)).total_seconds()))
              for item in start_times]

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

    # Add a check to make sure the altitudes actually cover the proper range.
    # Else, I will derive bad shifts
    bad_h = (alts.max()-ref_alt) < 0
    bad_m = (alts.min()-ref_alt) > 0
    if bad_h.any() or bad_m.any():
        if bad_h.any():
            logger.error('Ref. alt. sync impossible (MAX alt too low): %s',
                         [mid[0] for (ind, mid) in enumerate(prfs.get_info('mid'))
                          if bad_h.values[ind]])
        if bad_m.any():
            logger.error('Ref. alt. sync impossible (MIN alt too high): %s',
                         [mid[0] for (ind, mid) in enumerate(prfs.get_info('mid'))
                          if bad_m.values[ind]])

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

    out = [val if ~(bad_h | bad_m).values[ind] else None for (ind, val) in enumerate(out)]

    # Return the corresponding shifts, not forgetting that the first profile stays where it is.
    return list(out)


@log_func_call(logger)
def get_sync_shifts_from_val(prfs, max_shift=100, first_guess=None, valid_value_range=None,
                             sync_wrt_mid=None):
    """ Estimates the shifts required to synchronize profiles, such that <abs(val_A-val_B)> is
    minimized.

    Args:
        prfs (dvas.data.data.MultiRSProfiles): list of Profiles to compare
        max_shift (int, optional): maximum (absolute) shift to consider. Must be positive.
            Defaults to 100.
        first_guess (int|list of int, optional): starting guess around which to center the search.
            Defaults to None.
        valid_value_range (list, optional): if set, values outside the range set by this list of
            len(2) will be ignored.
        sync_wrt_mid (str, optional): if set, will sync all profiles with respect to this one.
            Defaults to 0 = first profile in the list.

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
        raise DvasError('first_guess should be the same length as prfs.')
    if sync_wrt_mid is None:
        sync_wrt_ind = 0
    else:
        sync_wrt_ind = prfs.get_info('mid').index([sync_wrt_mid])

    # In what follows, we assume that all the profiles are sampled with a fixed, uniform timestep.
    # Let's raise an error if this is not the case.
    # To do that, let's compute the differences between the different time steps, and check if it
    # is unique (or not) and identical for all profiles (or Not)
    ndts = prfs.get_prms(PRF_TDT).diff(periods=1, axis=0).nunique(axis=0, dropna=True)
    if np.any([item != 1 for item in ndts.values]):
        raise DvasError(f'The profiles do not all have uniform time steps: {ndts.values}')
    dts = [prfs.get_prms(PRF_TDT)[item][PRF_TDT].diff().unique().tolist()
           for item in range(len(prfs))]
    if np.any([item != dts[0] for item in dts]):
        raise DvasError(f'Inconsistent time steps between the different profiles: {dts}')

    # Let us first begin by extracting all the value arrays that need to be "cross-correlated".
    vals = prfs.get_prms(PRF_VAL)

    if valid_value_range is not None:
        if not isinstance(valid_value_range, list):
            raise DvasError(f'Bad data type for valid_value_range: {type(valid_value_range)}')
        if len(valid_value_range) != 2:
            raise DvasError(f'valid_value_range should be of len(2), not: {len(valid_value_range)}')

        assert valid_value_range[0] <= valid_value_range[1], \
            f'Odd valid_value_range limits: {valid_value_range}'

        # Hide the data, so it does not affect the derivation of sync shifts
        vals[vals < valid_value_range[0]] = np.nan
        vals[vals > valid_value_range[1]] = np.nan

    # Let's build an array of shifts to consider
    shifts = np.arange(-np.abs(max_shift), np.abs(max_shift)+1, 1)

    # Now, let's find the shifts that give rise to the smallest mean absolute offsets between the
    # profiles. If the shifts are larger than the array, I get a RunTime warning from numpy.
    # Let's just hide these away shamefully.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        ind = [[np.nanmean(np.abs(1. -
                                  vals.shift(first_guess[sync_wrt_ind])[sync_wrt_ind] /
                                  vals.shift(first_guess[ind]+shift)[ind]))
                for ind in range(len(prfs))] for shift in shifts]

    ind = np.nanargmin(np.array(ind), axis=0)

    # Issue some warning if I am at the very edge of the search range
    if np.any(max_shift-np.abs(shifts[ind]) < 10):
        logger.warning('sync_shift_from_val values is close from the edge of the search zone')

    # Return a list of shifts, resetting it to only have positive shifts.
    out = shifts[ind]+first_guess
    out = list(out-np.min(out))

    # Check if we are far from the first_guess ... and if so, raise a warning
    if any(np.abs(item - first_guess[i]) > 3 for i, item in enumerate(out)):
        logger.warning('Large offset from first guess: %s vs %s', out, first_guess)

    return out
