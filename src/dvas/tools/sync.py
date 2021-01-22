# -*- coding: utf-8 -*-
"""

Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains tools to synchronize profiles.

"""

# Import from other Python packages
import numpy as np

# Import from local modules
from ..logger import tools_logger as logger
from ..data.data import MultiRSProfile, MultiGDPProfile

def get_sync_shifts_from_alt(prfs, ref_alt=5000):
    """ A routine that estimates the shift required to synchronize profiles, based on their
    altitude index.

    Args:
        prfs (dvas.data.data.MultiRSProfiles): list of Profiles to compare
        ref_alt (float): which reference altitude to look for in the index

    Returns:
        list of int: list of shift required to synchronize profiles in order to match ref_alt

    """

    # Loop through the Profiles, and find which index best corresponds to the reference altitude
    ind = [np.abs(item.index.get_level_values('alt') - ref_alt).argmin()
           for item in prfs.get_prms()]

    # Return the corresponding shifts, not forgetting that the first profile stays where it is.
    return [0]+list(np.diff(ind))

def get_sync_shifts(prfs, **kwargs):
    """ Master function to measure shifts required to synchronize radiosonde profiles from a given
    flight.

    This function implements the official dvas recipe to derive the synchronization shifts.

    Args:
        prfs (dvas.data.data.MultiRSProfiles): list of Profiles to synchronize
        **kwargs: keyword arguments to be fed to the different underlying shift-identification
            routines.

    Returns:
        list of int: the official dvas shifts required to synchronize the profiles.
    """

    # 1) Let us start with a first crude guess of the necessary synchronization shift, by looking
    # at the measured altitudes
    guess_from_alt = get_sync_shifts_from_alt(prfs, **kwargs)

    # TODO: add other synchronization techniques to find the best possible shifts
    sync_shifts = guess_from_alt

    # 2) Let us re-scale all the shifts such that they are all positive and as small as possible.
    sync_shifts -= np.min(sync_shifts)

    return sync_shifts

def apply_sync_shifts(var_name, filt, sync_length, sync_shifts, is_gdp):
    """ Apply shifts to GDP and non-GDP profiles from a given flight, and upload them to the db.

    """

    # Let's apply the synchronization. Deal with GDPs and non-GDPs separately, to make sure the
    # uncertainties are being delt with accordingly

    # First the GDPs
    gdp_shifts = [item for (ind, item) in enumerate(sync_shifts) if is_gdp[ind]]
    gdps = MultiGDPProfile()
    gdps.load_from_db("and_({filt}, tag('gdp'))".format(filt=filt), var_name, 'tdtpros1',
                      alt_abbr='altpros1', ucr_abbr='treprosu_r', ucs_abbr='treprosu_s',
                      uct_abbr='treprosu_t')
    gdps.sort()
    gdps.rebase(sync_length, shift=gdp_shifts, inplace=True)
    gdps.save_to_db(add_tags=['sync'])

    # And now idem for the non-GDPs
    non_gdp_shifts = [item for (ind, item) in enumerate(sync_shifts) if not is_gdp[ind]]
    non_gdps = MultiRSProfile()
    non_gdps.load_from_db("and_({filt}, not_(tag('gdp')))".format(filt=filt), 'trepros1',
                          'tdtpros1', alt_abbr='altpros1')
    non_gdps.sort()
    non_gdps.rebase(sync_length, shift=non_gdp_shifts, inplace=True)
    non_gdps.save_to_db(add_tags=['sync'])



def sync_flight(evt_id, rig_id, **kwargs):
    """ High-level function responsible for synchronizing all the profile from a specific RS flight.

    This function directly synchronizes the profiles and upload them to the db with the 'sync' tag.

    Args:
        evt_id (str|int): event id to be synchronized
        rig_id (str|int): rig id to be synchronized
        **kwargs: keyword arguments to be fed to the underlying shift-identification routines.

    """

    # What search query will let me access the data I want ?
    filt = "and_(tag('e:{evt_id}'),tag('r:{rig_id}'),tag('raw'))".format(evt_id=evt_id,
                                                                         rig_id=rig_id)

    # Extract the data from the db
    prfs = MultiRSProfile()
    prfs.load_from_db(filt, 'trepros1', 'tdtpros1', alt_abbr='altpros1')
    prfs.sort()

    # Verify that that event datetime is actually the same for all the profiles. I will only
    # synchronize profiles that have flown together.
    dt_offsets = np.array([item.total_seconds() for item in np.diff(prfs.get_info('evt_dt'))])
    if any(dt_offsets > 0):
        logger.warning('Not all profiles to be synchronized have the same event_dt.')
        logger.warning('Offsets (w.r.t. first profile) in [s]: %s', dt_offsets)

    # Get the shift required for synchronization
    sync_shifts = get_sync_shifts(prfs, **kwargs)

    # Given the shifts, let's compute the length of the synchronized array. Do it such that no data
    # is actually cropped out, i.e. add NaN/NaT wherever needed.
    raw_lengths = [len(item.data) for item in prfs.profiles]
    sync_length = np.max(np.array(sync_shifts) + np.array(raw_lengths)) - \
                  np.min(sync_shifts)

    # Which of these profiles is a GDP ?
    is_gdp = ['gdp' in item for item in prfs.get_info('tags')]

    # Actually apply the shifts and update the db with the new profiles
    apply_sync_shifts('trepros1', filt, sync_length, sync_shifts, is_gdp)
