# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains tools to synchronize profiles.

"""

# Import from other Python packages
import numpy as np

# Import from this package
from ..logger import log_func_call
from ..logger import tools_logger as logger
from ..errors import DvasError

@log_func_call(logger)
def get_sync_shifts_from_alt(prfs, ref_alt=5000.):
    """ A routine that estimates the shifts required to synchronize profiles, based on the
    different in their altitude index.

    This is a very crude function that does the sync nased on a single altitude, and thus happily
    ignore any drift/stretch of any kind.

    Args:
        prfs (dvas.data.data.MultiRSProfiles): list of Profiles to compare
        ref_alt (float): the altitude at which to sync the profiles.

    Returns:
        list of int: list of shifts required to synchronize profiles in order to match ref_alt

    """

    # First, find the measured altitude that is closest to the reference altitude for the first
    # profile. Set this as the reference altitude, to make sure we match a real datapoint.
    ref_prf = prfs.profiles[0].data
    ind_ref_alt = np.abs(ref_prf.index.get_level_values('alt')-ref_alt).argmin()
    ref_alt = ref_prf.index.get_level_values('alt').values[ind_ref_alt]

    # Loop through the Profiles, and find which index best corresponds to the reference altitude
    ind = [np.abs(item.index.get_level_values('alt') - ref_alt).argmin()
           for item in prfs.get_prms()]

    # Return the corresponding shifts, not forgetting that the first profile stays where it is.
    return [0] + list(np.diff(ind))


@log_func_call(logger)
def get_sync_shifts_from_val(prfs, max_shift=100, first_guess=None):
    """ Estimates the shifts required to synchronize profiles, such that <abs(val_A-val_B)> is
    minimized.

    Args:
        prfs (dvas.data.data.MultiRSProfiles): list of Profiles to compare
        max_shift (int): maximum (absolute) shift to consider. Must be positive.
        first_guess (int|list of int): starting guess around which to center the search

    Returns:
        list of int: list of shifts required to synchronize profiles

    """

    # Start with some sanity checks
    if first_guess is None:
        first_guess = 0
    if isinstance(first_guess, int):
        first_guess = [first_guess] * len(prfs)
    if len(first_guess) != len(prfs):
        raise DvasError('Ouch ! first guess should be the same length as prfs.')

    # Let us first begin by extracting all the value arrays that need to be "cross-correlated".
    vals = [item.reset_index(drop=True, inplace=False) for item in prfs.get_prms('val')]

    # Let's build an array of shifts to consider
    shifts = np.arange(-np.abs(max_shift), np.abs(max_shift)+1, 1)

    # Now, let's find the shifts that give rise to the smallest mean absolute offsets between the
    # profiles
    ind = [np.array([np.nanmean(np.abs(vals[0]-item.shift(s))) for s in shifts]).argmin()
           for item in vals]

    # Issue some warning if I am at the very edge of the search range
    if np.any(np.abs(ind)-np.abs(max_shift) < 10):
        logger.warning('sync_shift_from_val values is close from the edge of the search zone')


    return shifts[ind]
