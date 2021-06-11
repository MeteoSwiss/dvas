"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level synchronization recipes for the UAII22 campaign
"""

# Import general Python packages
import numpy as np

# Import dvas modules and classes
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME
from dvas.logger import recipes_logger as logger
from dvas.logger import log_func_call
from dvas.data.data import MultiRSProfile, MultiGDPProfile
from dvas.tools import sync as dts

# Import from dvas_recipes
from .. import dynamic
from ..recipe import for_each_flight


@log_func_call(logger, time_it=True)
def apply_sync_shifts(var_name, filt, sync_length, sync_shifts, is_gdp):
    """ Apply shifts to GDP and non-GDP profiles from a given flight, and upload them to the db.

    Args:
        var_name (str): name of variable to sync, e.g. 'temp'
        filt (str): filtering query for the database.
        sync_length (int): length of the sync'ed profiles.
        sync_shifts (list of int): relative shifts required to sync the profiles.
        is_gdp (list of bool): to keep track of GDPs, in order to also sync their uncertainties.
    """

    # Let's apply the synchronization. Deal with GDPs and non-GDPs separately, to make sure the
    # uncertainties are being delt with accordingly

    # First the GDPs
    gdp_shifts = [item for (ind, item) in enumerate(sync_shifts) if is_gdp[ind]]
    gdps = MultiGDPProfile()
    gdps.load_from_db("and_({filt}, tags('gdp'))".format(filt=filt), var_name, 'time',
                      alt_abbr='gph',
                      ucr_abbr=dynamic.ALL_VARS[var_name]['ucr'],
                      ucs_abbr=dynamic.ALL_VARS[var_name]['ucs'],
                      uct_abbr=dynamic.ALL_VARS[var_name]['uct'],
                      ucu_abbr=dynamic.ALL_VARS[var_name]['ucu'])
    gdps.sort()
    gdps.rebase(sync_length, shifts=gdp_shifts, inplace=True)
    gdps.save_to_db(add_tags=['sync'])

    # And now idem for the non-GDPs
    non_gdp_shifts = [item for (ind, item) in enumerate(sync_shifts) if not is_gdp[ind]]
    non_gdps = MultiRSProfile()
    non_gdps.load_from_db("and_({filt}, not_(tags('gdp')))".format(filt=filt), var_name,
                          'time', alt_abbr='gph')
    non_gdps.sort()
    non_gdps.rebase(sync_length, shifts=non_gdp_shifts, inplace=True)
    non_gdps.save_to_db(add_tags=['sync'])


@for_each_flight
@log_func_call(logger, time_it=True)
def sync_flight(first_guess_var='temp'):
    """ Highest-level function responsible for synchronizing all the profile from a specific RS
    flight.

    This function directly synchronizes the profiles and upload them to the db with the 'sync' tag.

    Args:
        first_guess_var (str, optional): Name of the variable to use for getting a first guess of
            the synchronization shifts. Defaults to 'temp'.

    """

    # Extract the flight info
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    filt = "and_(tags('e:{}'), tags('r:{}'), tags('raw'))".format(eid, rid)

    # First, extract the temperature data from the db
    prfs = MultiRSProfile()
    prfs.load_from_db(filt, first_guess_var, dynamic.INDEXES[PRF_REF_TDT_NAME],
                      alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME])
    prfs.sort()

    # Get the Object IDs, so I can keep track of the different profiles and don't mess things up.
    oids = prfs.get_info(prm='oid')

    # Verify that that event datetime is actually the same for all the profiles. I should only
    # synchronize profiles that have flown together.
    dt_offsets = np.array([item.total_seconds() for item in np.diff(prfs.get_info('edt'))])
    if any(dt_offsets > 0):
        logger.warning('Not all profiles to be synchronized have the same event_dt.')
        logger.warning('Offsets (w.r.t. first profile) in [s]: %s', dt_offsets)

    # Get the preliminary shifts
    shifts_alt = dts.get_sync_shifts_from_alt(prfs)

    # Use this first guess to get a better set of shifts
    # TODO
    sync_shifts = shifts_alt

    # Given these shifts, let's compute the new length of the synchronized Profiles.
    # Do it such that no data is actually cropped out, i.e. add NaN/NaT wherever needed.
    raw_lengths = [len(item.data) for item in prfs.profiles]
    sync_length = np.max(np.array(sync_shifts) + np.array(raw_lengths)) - np.min(sync_shifts)

    # Which of these profiles is a GDP ?
    is_gdp = ['gdp' in item for item in prfs.get_info('tags')]

    # Keep track of the important info
    logger.info('oids: %s', oids)
    logger.info('is_gdp: %s', is_gdp)
    logger.info('sync_shifts: %s', sync_shifts)
    logger.info('sync_length: %s', sync_length)

    # Finally, apply the shifts and update the db with the new profiles, not overlooking the fact
    # that for GDPs, I also need to deal with the associated uncertainties.
    for var_name in dynamic.ALL_VARS:
        apply_sync_shifts(var_name, filt, sync_length, sync_shifts, is_gdp)