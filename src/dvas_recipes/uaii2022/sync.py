"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level synchronization recipes for the UAII2022 campaign
"""

# Import general Python packages
import logging
import numpy as np

# Import dvas modules and classes
from dvas.hardcoded import PRF_TDT, PRF_ALT, TAG_SYNC, TAG_GDP
from dvas.logger import log_func_call
from dvas.data.data import MultiRSProfile, MultiGDPProfile
from dvas.tools import sync as dts
from dvas.errors import DvasError

# Import from dvas_recipes
from .. import dynamic
from ..recipe import for_each_flight
from ..errors import DvasRecipesError
from . import tools
from .. import utils as dru

# Setup local logger
logger = logging.getLogger(__name__)


@log_func_call(logger, time_it=False)
def apply_sync_shifts(var_name, filt, sync_length, sync_shifts, is_gdp):
    """ Apply shifts to GDP and non-GDP profiles from a given flight, and upload them to the db.

    Args:
        var_name (str): name of variable to sync, e.g. 'temp'
        filt (str): filtering query for the database.
        sync_length (int): length of the sync'ed profiles.
        sync_shifts (list of int): relative shifts required to sync the profiles.
            Convention: row n become row n+shift.
        is_gdp (list of bool): to keep track of GDPs, in order to also sync their uncertainties.
    """

    # Let's apply the synchronization. Deal with GDPs and non-GDPs separately, to make sure the
    # uncertainties are being dealt with accordingly

    # First the GDPs
    gdp_shifts = [item for (ind, item) in enumerate(sync_shifts) if is_gdp[ind]]
    # Let's force users to have GDPs. That's really what dvas is meant for ...
    if len(gdp_shifts) == 0:
        raise DvasRecipesError('Ouch ! No GDPs to sync ?!')

    gdps = MultiGDPProfile()
    gdps.load_from_db("and_({}, tags('{}'))".format(filt, TAG_GDP), var_name,
                      dynamic.INDEXES[PRF_TDT],
                      alt_abbr=dynamic.INDEXES[PRF_ALT],
                      ucr_abbr=dynamic.ALL_VARS[var_name]['ucr'],
                      ucs_abbr=dynamic.ALL_VARS[var_name]['ucs'],
                      uct_abbr=dynamic.ALL_VARS[var_name]['uct'],
                      ucu_abbr=dynamic.ALL_VARS[var_name]['ucu'])
    logger.info('Loaded %i GDP profiles for variable %s.', len(gdps), var_name)
    gdps.sort()
    gdps.rebase(sync_length, shifts=gdp_shifts, inplace=True)
    gdps.save_to_db(
        add_tags=[TAG_SYNC, dynamic.CURRENT_STEP_ID],
        rm_tags=dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID)
        )

    # And now idem for the non-GDPs
    non_gdp_shifts = [item for (ind, item) in enumerate(sync_shifts) if not is_gdp[ind]]
    # Only proceed if some non-GDP profiles were found. This makes pure-GDP flights possible.
    if len(non_gdp_shifts) > 0:
        non_gdps = MultiRSProfile()
        non_gdps.load_from_db("and_({}, not_(tags('{}')))".format(filt, TAG_GDP), var_name,
                              dynamic.INDEXES[PRF_TDT],
                              alt_abbr=dynamic.INDEXES[PRF_ALT])
        logger.info('Loaded %i non-GDP profiles for variable %s.', len(non_gdps), var_name)
        non_gdps.sort()
        non_gdps.rebase(sync_length, shifts=non_gdp_shifts, inplace=True)
        non_gdps.save_to_db(
            add_tags=[TAG_SYNC, dynamic.CURRENT_STEP_ID],
            rm_tags=dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID)
            )


@for_each_flight
@log_func_call(logger, time_it=True)
def sync_flight(start_with_tags, anchor_alt, global_match_var):
    """ Highest-level function responsible for synchronizing all the profile from a specific RS
    flight.

    This function directly synchronizes the profiles and upload them to the db with the 'sync' tag.

    Args:
        start_with_tags (str|list): list of tags to identify profiles to sync in the db.
        anchor_alt (int|float): (single) altitude around which to anchor all profiles.
            Used as a first - crude!- guess to get the biggest shifts out of the way.
            Relies on dvas.tools.sync.get_synch_shifts_from_alt()
        global_match_var (str): Name of the variable to use for getting the synchronization shifts
            from a flobal match of the profile.
            Relies on dvas.tools.sync.get_synch_shifts_from_val()

    """

    # Format the tags
    tags = dru.format_tags(start_with_tags)

    # Extract the flight info
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    filt = tools.get_query_filter(tags_in=tags+[eid, rid], tags_out=dru.rsid_tags(pop=tags))

    # First, extract the RS profiles from the db, for the requested variable
    prfs = MultiRSProfile()
    prfs.load_from_db(filt, global_match_var, dynamic.INDEXES[PRF_TDT],
                      alt_abbr=dynamic.INDEXES[PRF_ALT])
    prfs.sort()

    # Get the Object IDs, so I can keep track of the different profiles and don't mess things up.
    oids = prfs.get_info(prm='oid')

    # Verify that that event datetime is actually the same for all the profiles. I should only
    # synchronize profiles that have flown together.
    dt_offsets = np.array([item.total_seconds() for item in np.diff(prfs.get_info('edt'))])
    if any(dt_offsets > 0):
        logger.error('Not all profiles to be synchronized have the same event_dt.')
        logger.error('Offsets (w.r.t. first profile) in [s]: %s', dt_offsets)

    # Get the preliminary shifts from the altitude
    shifts_alt = dts.get_sync_shifts_from_alt(prfs, ref_alt=anchor_alt)
    logger.info('sync. shifts from alt (%.1f): %s', anchor_alt, shifts_alt)

    # Use these to get synch shifts from the variable
    shifts_val = dts.get_sync_shifts_from_val(prfs, max_shift=100, first_guess=shifts_alt)
    logger.info('Sync. shifts from "%s": %s', global_match_var, shifts_val)

    # Get shifts from the GPS times
    try:
        shifts_gps = dts.get_sync_shifts_from_starttime(prfs)
        logger.info('Sync. shifts from starttime: %s', shifts_gps)
    except DvasError as err:
        logger.critical(err)

    # Use these as best synch shifts
    sync_shifts = shifts_val

    # Given these shifts, let's compute the new length of the synchronized Profiles.
    # Do it such that no data is actually cropped out, i.e. add NaN/NaT wherever needed.
    raw_lengths = [len(item.data) for item in prfs.profiles]
    sync_length = np.max(np.array(sync_shifts) + np.array(raw_lengths)) - np.min(sync_shifts)

    # Which of these profiles is a GDP ?
    is_gdp = prfs.has_tag(TAG_GDP)

    # Keep track of the important info
    logger.info('oids: %s', oids)
    logger.info('mids: %s', prfs.get_info(prm='mid'))
    logger.info('is_gdp: %s', is_gdp)
    logger.info('sync_shifts: %s', sync_shifts)
    logger.info('sync_length: %s', sync_length)

    # Finally, apply the shifts and update the db with the new profiles, not overlooking the fact
    # that for GDPs, I also need to deal with the associated uncertainties.
    for var_name in dynamic.ALL_VARS:
        logger.info('Applying sync shifts for variable: %s', var_name)
        apply_sync_shifts(var_name, filt, sync_length, sync_shifts, is_gdp)
