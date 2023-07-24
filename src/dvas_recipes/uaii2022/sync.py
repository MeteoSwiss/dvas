"""
Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level synchronization recipes for the UAII2022 campaign
"""

# Import general Python packages
import logging
from datetime import timedelta
import numpy as np

# Import dvas modules and classes
from dvas.hardcoded import PRF_TDT, PRF_ALT, TAG_SYNC, TAG_GDP, MTDTA_FIRST
from dvas.logger import log_func_call
from dvas.data.data import MultiRSProfile, MultiGDPProfile
from dvas.tools import sync as dts
from dvas.errors import DvasError

# Import from dvas_recipes
from ..errors import DvasRecipesError
from .. import dynamic
from ..recipe import for_each_flight
from . import tools
from .. import utils as dru

# Setup local logger
logger = logging.getLogger(__name__)


@log_func_call(logger, time_it=False)
def adjust_first_timestamp(prfs, sync_shifts, mtdta_key=MTDTA_FIRST):
    """ Apply the sync_shifts to the first_timstamp stored in the metadata.

    Args:
        prfs: RSProfiles, GDPProfiles, or CWSProfiles.
        sync_shifts (list of int): the sync shifts to apply.
        mtdta_key (str, optional): the metadata key to adjust. Defaults to MTDTA_FIRST.

    This routine fixes #256.

    """

    assert len(prfs) == len(sync_shifts), f'length mismatch: {len(prfs)} vs {len(sync_shifts)}'

    # Let's loop throught the shifts
    for (s_ind, s_val) in enumerate(sync_shifts):

        # First, compute the time step
        t_step = np.unique(np.diff(prfs[s_ind].data.index.get_level_values('tdt').seconds))
        t_step = [val for val in t_step if not np.isnan(val)]
        if len(t_step) != 1:
            raise DvasRecipesError(f'Cannot compute time shift with t_step: {t_step}')
        t_step = t_step[0]

        # Compute the time shift. Assume/remember than negative shifts are cropped, while positive
        # ones are padded with NaN's (i.e. 0 = shift reference)
        t_shift = -s_val*t_step

        logger.debug('t_step = %.2f sec (%s)', t_step, ','.join(prfs[s_ind].info.mid))
        logger.debug('s_val = %+i steps', s_val)
        logger.debug('%s actual shift: %+.2f sec (%s)', mtdta_key, t_shift,
                     ','.join(prfs[s_ind].info.mid))

        # Update the metadata field as required
        prfs[s_ind].info.add_metadata(
            mtdta_key, prfs[s_ind].info.metadata[mtdta_key] + timedelta(seconds=float(t_shift)))

        # Add also the computed sjift to the metadata
        prfs[s_ind].info.add_metadata('sync_shift_applied', f"{s_val:+} steps")
        prfs[s_ind].info.add_metadata(f'sync_shift_correction_to_{mtdta_key}',
                                      f"{t_shift:+.1f} sec")


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
    if len(gdp_shifts) > 0:
        gdps = MultiGDPProfile()
        gdps.load_from_db(f"and_({filt}, tags('{TAG_GDP}'))", var_name,
                          dynamic.INDEXES[PRF_TDT],
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          ucs_abbr=dynamic.ALL_VARS[var_name]['ucs'],
                          uct_abbr=dynamic.ALL_VARS[var_name]['uct'],
                          ucu_abbr=dynamic.ALL_VARS[var_name]['ucu'])
        logger.info('Loaded %i GDP profiles for variable %s.', len(gdps), var_name)
        gdps.sort()
        gdps.rebase(sync_length, shifts=gdp_shifts, inplace=True)
        adjust_first_timestamp(gdps, gdp_shifts, mtdta_key=MTDTA_FIRST)
        gdps.save_to_db(
            add_tags=[TAG_SYNC, dynamic.CURRENT_STEP_ID],
            rm_tags=dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID)
            )

    # And now idem for the non-GDPs
    non_gdp_shifts = [item for (ind, item) in enumerate(sync_shifts) if not is_gdp[ind]]
    # Only proceed if some non-GDP profiles were found. This makes pure-GDP flights possible.
    if len(non_gdp_shifts) > 0 and var_name not in ['lat', 'lon']:
        non_gdps = MultiRSProfile()
        non_gdps.load_from_db(f"and_({filt}, not_(tags('{TAG_GDP}')))", var_name,
                              dynamic.INDEXES[PRF_TDT],
                              alt_abbr=dynamic.INDEXES[PRF_ALT])
        logger.info('Loaded %i non-GDP profiles for variable %s.', len(non_gdps), var_name)
        non_gdps.sort()
        non_gdps.rebase(sync_length, shifts=non_gdp_shifts, inplace=True)
        adjust_first_timestamp(non_gdps, non_gdp_shifts, mtdta_key=MTDTA_FIRST)
        non_gdps.save_to_db(
            add_tags=[TAG_SYNC, dynamic.CURRENT_STEP_ID],
            rm_tags=dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID)
            )


@for_each_flight
@log_func_call(logger, time_it=False)
def sync_flight(start_with_tags, anchor_alt, global_match_var, valid_value_range, sync_wrt_mid,
                crop_pre_gdp):
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
        valid_value_range (list|None): if a len(2) list is provided, values of the global_match_var
            outside this range will be ignored when deriving the global-match synch shifts.
        sync_wrt_mid (str): radiosonde model-id against which to synchronize profiles.
        crop_pre_gdp (bool): if True, any data taken before gdp values excists will be
            cropped.

    """

    # Format the tags
    tags = dru.format_tags(start_with_tags)

    # Extract the flight info
    (_, eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    filt = tools.get_query_filter(tags_in=tags+[eid, rid], tags_out=None)

    # First, extract the RS profiles from the db, for the requested variable
    prfs = MultiRSProfile()
    prfs.load_from_db(filt, global_match_var, dynamic.INDEXES[PRF_TDT],
                      alt_abbr=dynamic.INDEXES[PRF_ALT])
    prfs.sort()

    # Which of these profiles is a GDP ?
    is_gdp = prfs.has_tag(TAG_GDP)

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
    logger.info('Sync. shifts from alt (%.1f): %s', anchor_alt, shifts_alt)

    if any([item is None for item in shifts_alt]):
        raise DvasRecipesError('Invalid shift value derived from alt')

    # Use these to get synch shifts from the variable
    shifts_val = dts.get_sync_shifts_from_val(prfs, max_shift=20, first_guess=shifts_alt,
                                              valid_value_range=valid_value_range,
                                              sync_wrt_mid=sync_wrt_mid)
    logger.info('Sync. shifts from "%s": %s', global_match_var, shifts_val)

    # Get shifts from the GNSS times
    try:
        shifts_gnss = dts.get_sync_shifts_from_starttime(prfs)
        logger.info('Sync. shifts from GNSS starttime: %s', shifts_gnss)
    except DvasError as err:
        logger.critical(err)

    # GNSS times have many issues ... let's use gph sync shifts instead.
    sync_shifts = shifts_val

    # Now, let's reset the shift depening on whether we want to crop pre-GDP data, or not
    if crop_pre_gdp:
        shift_min = np.min([s_val for (s_ind, s_val) in enumerate(sync_shifts) if is_gdp[s_ind]])
    else:
        shift_min = np.min(sync_shifts)
    sync_shifts -= shift_min

    # Given these shifts, let's compute the new length of the synchronized Profiles.
    # Do it such that no data is actually cropped out, i.e. add NaN/NaT wherever needed.
    orig_lengths = [len(item.data) for item in prfs.profiles]
    sync_length = np.max(np.array(sync_shifts) + np.array(orig_lengths))

    # Keep track of the important info
    logger.info('oids: %s', oids)
    logger.info('mids: %s', prfs.get_info(prm='mid'))
    logger.info('is_gdp: %s', is_gdp)
    logger.info('crop_pre_gdp: %s', crop_pre_gdp)
    logger.info('sync_shifts: %s', sync_shifts)
    logger.info('sync_length: %s', sync_length)

    # Finally, apply the shifts and update the db with the new profiles, not overlooking the fact
    # that for GDPs, I also need to deal with the associated uncertainties.
    for var_name in dynamic.ALL_VARS:
        # I do not have any wind vector yet ...
        if var_name == 'wvec':
            continue
        logger.info('Applying sync shifts for variable: %s', var_name)
        apply_sync_shifts(var_name, filt, sync_length, sync_shifts, is_gdp)
