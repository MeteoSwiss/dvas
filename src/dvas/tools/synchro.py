# -*- coding: utf-8 -*-
"""

Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains tools to synchronize profiles.

"""

# Import from other Python packages
from datetime import datetime
import numpy as np

# Import from local modules
from ..logger import tools_logger as logger
from ..data.data import MultiRSProfile, MultiGDPProfile

def get_shifts_from_alt(prfs, ref_alt=5000):
    """ A routine that estimates the shift required to synchronize profiles, based on their
    altitude index.

    Args:
        prfs (dvas.data.data.MultiRSProfiles): list of Profiles to compare
        ref_alt (float): which reference altitude to look for in the index

    Returns:
        list of int: list of shift required to synchronize profiles in order to match ref_alt

    """

    # Extract the list of DataFrames
    prf_list = prfs.get_prms()

    # Loop through these, and find which index best corresponds to the reference altitude
    ind = [np.abs(item.index.get_level_values('alt') - ref_alt).argmin()
           for item in prfs.get_prms()]

    # Return the corresponding shifts, not forgetting that the first profile stays where it is.
    return [0]+list(np.diff(ind))

def synchronize_rs_flight(evt_id, rig_id, vars, ref_var='trepros1', ref_alt=5000):
    """ High-level function responsible for synchronizing all the profile from a specific RS flight.

    Args:
        evt_id (str|int): event id to be synchronized
        rig_id (str|int): rig id to be synchronized
        vars (list of str): vars to be synchronized
        ref_vars (str, optional): name of variable to use for inferring the proper synchronization
            parameters. Defaults to 'trepros1'.
        ref_alt (float, optional): the reference altitude for deriving a crude guess of the shifts
            between profiles, based on their respective altitude arrays.

    Returns:

    """

    # Step 1: We need to derive the synchronization.

    # What search query will let me access the data I want ?
    filt = "and_(tag('e:{evt_id}'),tag('r:{rig_id}'),tag('raw'))".format(evt_id=evt_id,
                                                                         rig_id=rig_id)

    # Extract the data from the db
    prfs = MultiRSProfile()
    prfs.load_from_db(filt, ref_var, 'tdtpros1', alt_abbr='altpros1')
    prfs.sort()

    # Verify that that event datetime is actually the same for all the profiles. I will only
    # synchronize profiles that have flown together.
    dt_offsets = np.array([item.total_seconds() for item in np.diff(prfs.get_info('evt_dt'))])
    if any(dt_offsets > 0):
        logger.warning('Not all profiles to be synchronized have the same event_dt.')
        logger.warning('Offsets (w.r.t. first profile) in [s]: %s', dt_offsets)

    # Let us start with a first crude guess of the necessary synchronization shift, by looking
    # at the measured altitudes
    guess_from_alt = get_shifts_from_alt(prfs, ref_alt=ref_alt)
    print('guess_from_alt:', guess_from_alt)

    # TODO: add other synchronization techniques to find the best possible shifts
    synchro_shifts = guess_from_alt

    # Let us re-scale all the shifts such that they all become positive and as small as possible.
    synchro_shifts -= np.min(synchro_shifts)

    # Given the shifts, let's compute the length of the synchronized array. Do it such that no data is
    # is actually cropped out, i.e. add NaN/NaT wherever needed.
    raw_lengths = [len(item.data) for item in prfs.profiles]
    synchro_length = np.max(np.array(guess_from_alt)+np.array(raw_lengths)) - \
                     np.min(guess_from_alt)

    # Which of these is a GDP ?
    is_gdp = ['gdp' in item for item in prfs.get_info('tags')]

    # Let's apply the synchronization. Deal with GDPs and non-GDPs separately, to make sure the
    # uncertainties are being delt with accordingly
    # TODO: put this into a dedicated function, also as a function of the variable.

    # First the GDPs
    gdp_shifts = [item for (ind, item) in enumerate(synchro_shifts) if is_gdp[ind]]
    gdps = MultiGDPProfile()
    gdps.load_from_db("and_({filt}, tag('gdp'))".format(filt=filt), ref_var, 'tdtpros1',
                      alt_abbr='altpros1', ucr_abbr='treprosu_r', ucs_abbr='treprosu_s',
                      uct_abbr='treprosu_t')
    gdps.sort()
    gdps.rebase(synchro_length, shift=gdp_shifts, inplace=True)
    gdps.save_to_db(add_tags=['sync'])

    # And now idem for the non-GDPs
    non_gdp_shifts = [item for (ind, item) in enumerate(synchro_shifts) if not is_gdp[ind]]
    non_gdps = MultiRSProfile()
    non_gdps.load_from_db("and_({filt}, not_(tag('gdp')))".format(filt=filt), ref_var, 'tdtpros1',
                          alt_abbr='altpros1')
    non_gdps.sort()
    non_gdps.rebase(synchro_length, shift=non_gdp_shifts, inplace=True)
    non_gdps.save_to_db(add_tags=['sync'])
