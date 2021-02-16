# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains tools to synchronize profiles.

"""

# Import from other Python packages
import numpy as np

def get_sync_shifts_from_alt(prfs, ref_alt=5000):
    """ A routine that estimates the shifts required to synchronize profiles, based on the
    different altitude index.

    Args:
        prfs (dvas.data.data.MultiRSProfiles): list of Profiles to compare
        ref_alt (float): which reference altitude to look for in the index

    Returns:
        list of int: list of shifts required to synchronize profiles in order to match ref_alt

    """

    # Loop through the Profiles, and find which index best corresponds to the reference altitude
    ind = [np.abs(item.index.get_level_values('alt') - ref_alt).argmin()
           for item in prfs.get_prms()]

    # Return the corresponding shifts, not forgetting that the first profile stays where it is.
    return [0] + list(np.diff(ind))

def get_sync_shifts_from_crosscor(prfs):
    """ Estimates the shifts required to synchronize profiles, based on their cross correlation.

    """
