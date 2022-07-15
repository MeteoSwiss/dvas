"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: basic high-level recipes for the UAII2022 campaign
"""

# Import general Python packages
import logging
import pandas as pd

# Import dvas modules and classes
# from dvas.logger import recipes_logger as logger
from dvas.environ import path_var
from dvas.logger import log_func_call
from dvas.data.data import MultiRSProfile, MultiGDPProfile
from dvas.hardcoded import PRF_TDT, PRF_ALT, PRF_IDX
from dvas.hardcoded import TAG_GDP, TAG_CLN, FLG_DESCENT, MTDTA_BPT
from dvas.dvas import Database as DB

# Import from dvas_recipes
from .. import dynamic
from ..recipe import for_each_flight, for_each_var
from ..errors import DvasRecipesError
from . import tools
from .. import utils as dru

logger = logging.getLogger(__name__)


@log_func_call(logger, time_it=False)
def prf_summary(create_eid_edt_files=False):
    """ Exports a summary of the different profiles in the DB.

    Args:
        create_eid_edt_files (bool, optional): if True, will issue a set of empty files that tie
            eids to edts. Used for the UAII 2022 preview tool. Defaults to False.
    """

    view = DB.extract_global_view()

    # Save this file to csv
    fn_out = path_var.output_path / (dynamic.CURRENT_STEP_ID + '_profile_list.csv')
    view.to_csv(fn_out, index=False)
    logger.info('Created profile list: %s', fn_out)

    # The following files are required for the quick preview visualization tool developed for the
    # UAII 2022.
    if create_eid_edt_files:
        for _, row in view.loc[:, ['edt', 'eid']].drop_duplicates().iterrows():

            fn_out = path_var.output_path / (dynamic.CURRENT_STEP_ID +
                                             f"_eid-edt_{row['eid'].replace(':', '')}" +
                                             f"_{row['edt'].strftime('%Y%m%dT%H%M%S')}")
            open(fn_out, 'a').close()


@log_func_call(logger, time_it=False)
def flag_descent(prfs):
    """ Set a dedicated flag for any point beyond the burst point.

    Args:
        prfs (MultiRSProfile|MultiGDPProfile): the profiles to flag (individually).

    Note:
        If available, this function will use the metadata info to set the flags. Else, it will
        simply flag any point beyond the max altitude.

    Important:
        This function assumes that the profiles have not been shifted in any way (yet) !

    Returns:
        MultiRSProfile|MultiGDPProfile: the flagged profiles.

    """

    # Loop through each profile, and figure out if I need to flag anything
    for prf in prfs:

        # Check if a bpt_time is avalaible. Else, look for the max altitude reached.
        if MTDTA_BPT not in prf.info.metadata.keys():
            logger.warning('"%s" not found in metadata for: %s', MTDTA_BPT, prf.info.src)
            use_max = True
        elif prf.info.metadata[MTDTA_BPT] is None:
            logger.warning('"%s" is None for: %s', MTDTA_BPT, prf.info.src)
            use_max = True
        else:
            logger.info('"%s" ok for: %s', MTDTA_BPT, prf.info.src)
            use_max = False

        if use_max:
            max_alt_id = prf.data.index.get_level_values(PRF_ALT).argmax()
            which = prf.data.index.get_level_values(PRF_IDX) > max_alt_id
            logger.info('Points after max alt %.1f [%s] @ %.1f [s] flagged as "%s".',
                        prf.data.index[max_alt_id][1],
                        prfs.var_info[PRF_ALT]['prm_unit'],
                        prf.data.index[max_alt_id][2].total_seconds(),
                        FLG_DESCENT)

        else:
            # Extract the burst point, and try to convert it to a time delta
            bpt_time = (prf.info.metadata[MTDTA_BPT]).split(' ')
            if len(bpt_time) != 2:
                raise DvasRecipesError(
                    f'Ouch ! "{MTDTA_BPT}" is weird: {prf.info.metadata["bpt_time"]}')

            bpt_time = pd.Timedelta(float(bpt_time[0]), bpt_time[1])
            which = prf.data.index.get_level_values(PRF_TDT) >= bpt_time
            logger.info('Points after burst point @ %s [s] flagged as "%s"',
                        bpt_time.total_seconds(), FLG_DESCENT)

        prf.set_flg(FLG_DESCENT, True, index=which)

    return prfs


@log_func_call(logger, time_it=False)
def cleanup_steps(prfs, resampling_freq, interp_dist, crop_descent, timeofday=None):
    """ Execute a series of cleanup-steps common to GDP and non-GDP profiles. This function is here
    to avoid duplicating code. The cleanup-up profiles are directly saved to the DB with the tag:
    TAG_CLN

    Args:
        prfs (MultiRSProfile|MultiGDPProfile): the profiles to cleanup.
        resampling_freq (str): time step frequency, to feed :py:func:`pandas.timedelta_range`, e.g.
            '1s'.
        interp_dist (float|int): distance to the nearest real measurement, in s, beyond which a
            resampled point is forced to NaN (i.e. "dvas does not interpolate !")
        crop_descent (bool): if True, and data with the flag "descent" will be cropped out for good.
        timeofday (str): if set, will tag the Profile with this time of day. Defaults to None.

    """

    # Flag descent data (do this *after* the resampling so I do not need to worry about it)
    prfs = flag_descent(prfs)

    # Crop the descent data if warranted
    if crop_descent:
        for (ind, prf) in enumerate(prfs):
            prfs[ind].data = prf.data.loc[~prf.has_flg(FLG_DESCENT)]

    # Add the TimeOfDay tag, if warranted
    if timeofday is not None:
        for (ind, prf) in enumerate(prfs):
            if not prf.has_tag(timeofday):
                prf.info.add_tags(timeofday)
                logger.info('Adding missing TimeOfDay tag to %s profile.', prf.info.mid)

    # Resample the profiles as required
    prfs.resample(freq=resampling_freq, interp_dist=interp_dist, inplace=True,
                  chunk_size=dynamic.CHUNK_SIZE, n_cpus=dynamic.N_CPUS)

    # Save back to the DB
    prfs.save_to_db(
        add_tags=[TAG_CLN, dynamic.CURRENT_STEP_ID],
        rm_tags=dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID)
        )


@for_each_var
@for_each_flight
@log_func_call(logger, time_it=True)
def cleanup(start_with_tags, fix_gph_uct=None, check_tropopause=False, **args):
    """ Highest-level function responsible for doing an initial cleanup of the data.

    Args:
        start_with_tags (str|list): list of tags to identify profiles to clean in the db.
        fix_gph_uct (list, optional): list of mid values for which to correct NaN values (see #205).
            Defaults to None.
        check_tropopause (bool, optional): if True, will compare the dvas tropopause to the GRUAN
            one. Defaults to False. Raises a log error if the dvas measure is >20m off.
        **args: arguments to be fed to :py:func:`.cleanup_steps`.

    """

    # Format the tags
    tags = dru.format_tags(start_with_tags)

    # Extract the flight info
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    filt = tools.get_query_filter(tags_in=tags + [eid, rid],
                                  tags_out=dru.rsid_tags(pop=tags))

    # Let's extract the summary of what the DB contains
    db_view = DB.extract_global_view()
    this_flight = (db_view.eid == eid) * (db_view.rid == rid)

    # Check the time of day for the flight, see if it is consistent.
    timeofday = db_view[this_flight].tod[db_view[this_flight].tod.notna()].unique()
    if len(timeofday) == 1:
        timeofday = timeofday[0]
        logger.info('TimeOfDay for flight (%s, %s): %s', eid, rid, timeofday)
    elif len(timeofday) == 0:
        logger.error('TimeOfDay unknown for flight: %s, %s', eid, rid)
        timeofday = None
    else:
        raise DvasRecipesError(
            f'TimeOfDay tags inconsistent for flight ({eid}, {rid}): {timeofday}')

    # I need to treat GDPs and non-GDPs separately, since the former have uncertainties that also
    # need to be cleaned accordingly.

    # Start with the GDPs
    if db_view[this_flight].is_gdp.any():
        logger.info('Cleaning GDP profiles for flight %s and variable %s',
                    dynamic.CURRENT_FLIGHT,
                    dynamic.CURRENT_VAR)

        gdp_prfs = MultiGDPProfile()
        gdp_prfs.load_from_db(f'and_({filt}, tags("{TAG_GDP}"))', dynamic.CURRENT_VAR,
                              tdt_abbr=dynamic.INDEXES[PRF_TDT],
                              alt_abbr=dynamic.INDEXES[PRF_ALT],
                              ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                              ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                              uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                              ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                              inplace=True)

        logger.info('Loaded %i GDP profiles from the DB.', len(gdp_prfs))

        # Deal with the faulty RS41 gph_uc_tcor values (NaNs when they should not be, see #205)
        if dynamic.CURRENT_VAR == 'gph' and fix_gph_uct is not None:
            # Safety check
            if not isinstance(fix_gph_uct, list):
                raise DvasRecipesError(f'Ouch ! fix_gph_uct should be a list, not: {fix_gph_uct}')

            # Start looping through the profiles, to identify the ones I need.
            for gdp in gdp_prfs:
                if gdp.info.mid[0] in fix_gph_uct:  # Here we assume that each GDP has a single mid

                    # What are the bug conditions ?
                    cond1 = gdp.data.loc[:, 'uct'].isna()
                    cond2 = ~gdp.data.loc[:, 'val'].isna()

                    if (n_bad := (cond1 & cond2).sum()) > 0:  # True = 1
                        logger.info('Fixing %i bad gph_uct values for mid: %s',
                                    n_bad, gdp.info.mid[0])
                        # Fix the bug
                        gdp.data.loc[cond1 & cond2, 'uct'] = gdp.data.loc[:, 'uct'].max(skipna=True)

        # Validate the GRUAN tropopause calculation
        if dynamic.CURRENT_VAR == 'temp' and check_tropopause:
            logger.info('Comparing the GRUAN vs dvas tropopause:')

            for gdp_prf in gdp_prfs:
                if 'gruan_tropopause' not in gdp_prf.info.metadata.keys():
                    logger.warning('No GRUAN tropopause found in %s', gdp_prf.info.src)
                    continue

                # Let's compute the dvas tropopause
                dvas_trop = tools.find_tropopause(gdp_prf, algo='gruan')

                # Raise a log error if we are more than 20m off, or if the format is not understood.
                match gdp_prf.info.metadata['gruan_tropopause'].split(' '):
                    case [val, 'gpm']:

                        msg = 'Tropopause - GRUAN: {} [m] vs {} [m] :dvas ({})'.format(
                            val, dvas_trop[1], gdp_prf.info.src)

                        if abs(float(val)-dvas_trop[1]) <= 20:
                            logger.info(msg)
                        else:
                            logger.error(msg)

                    case _:
                        logger.error('Unknown GRUAN tropopause format: %s',
                                     gdp_prf.info.metadata['gruan_tropopause'])

        # Now launch more generic cleanup steps
        cleanup_steps(gdp_prfs, **args, timeofday=timeofday)

    # Process the non-GDPs, if any
    if not db_view[this_flight].is_gdp.all():
        logger.info('Cleaning non-GDP profiles for flight %s and variable %s',
                    dynamic.CURRENT_FLIGHT,
                    dynamic.CURRENT_VAR)

        # Extract the data from the db
        rs_prfs = MultiRSProfile()
        rs_prfs.load_from_db(f'and_({filt}, not_(tags("{TAG_GDP}")))',
                             dynamic.CURRENT_VAR, dynamic.INDEXES[PRF_TDT],
                             alt_abbr=dynamic.INDEXES[PRF_ALT])

        logger.info('Loaded %i RS profiles from the DB.', len(rs_prfs))

        cleanup_steps(rs_prfs, **args, timeofday=timeofday)
