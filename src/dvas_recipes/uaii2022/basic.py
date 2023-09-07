"""
Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

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
from dvas.hardcoded import PRF_TDT, PRF_ALT, PRF_IDX, PRF_VAL
from dvas.hardcoded import MTDTA_FIRST, MTDTA_LAUNCH, MTDTA_BURST
from dvas.hardcoded import TAG_GDP, TAG_CLN, FLG_PRELAUNCH, FLG_ASCENT, FLG_DESCENT
from dvas.hardcoded import FLG_ISINVALID, FLG_WASINVALID
from dvas.dvas import Database as DB

# Import from dvas_recipes
from .. import dynamic
from ..recipe import for_each_flight, for_each_var
from ..errors import DvasRecipesError
from . import tools
from .. import utils as dru

logger = logging.getLogger(__name__)


@log_func_call(logger, time_it=False)
def prf_summary():
    """ Exports a summary of the different profiles in the DB.

    Args:

    """

    view = DB.extract_global_view()

    # Save this file to csv
    fn_out = path_var.output_path / (dynamic.CURRENT_STEP_ID + '_profile_list.csv')
    view.to_csv(fn_out, index=False)
    logger.info('Created profile list: %s', fn_out)


@log_func_call(logger, time_it=False)
def flag_phases(prfs):
    """ Flag the different profile phases, i.e. prelaunch, ascent, and descent.

    Args:
        prfs (MultiRSProfile|MultiGDPProfile): the profiles to flag (individually).

    If available, this function will use the metadata info to set the flags.
    If no first or burst timestamp is defined, any point beyond the max altitude will be flagged as
    descent.
    If no first or launch timestamp is set, the profile is assumed to start at the launch time.

    Important:
        This function assumes that the profiles have not been shifted in any way (yet) !

    Returns:
        MultiRSProfile|MultiGDPProfile: the flagged profiles.

    """

    # Loop through each profile, and figure out if I need to flag anything
    for prf in prfs:

        # First, look for preflight data
        if all((item in prf.info.metadata.keys()) and
               ((item, None) not in prf.info.metadata.items())
               for item in [MTDTA_FIRST, MTDTA_LAUNCH]):

            prelaunch_ends_at = prf.info.metadata[MTDTA_LAUNCH] - prf.info.metadata[MTDTA_FIRST]
            if prelaunch_ends_at.total_seconds() != 0:
                if prelaunch_ends_at.total_seconds() < 0:
                    logger.warning('first_timestamp > launch_timestamp (%s)', prf.info.src)
            else:
                logger.debug('No prelaunch data identified (%s)', prf.info.src)
        else:
            logger.warning('Cannot identify pre-launch phase: missing metadata (%s)', prf.info.src)
            prelaunch_ends_at = pd.Timedelta(0, 's')

        is_prelaunch = prf.data.index.get_level_values(PRF_TDT) < prelaunch_ends_at

        # Then look for descent data
        if all((item in prf.info.metadata.keys()) and
               ((item, None) not in prf.info.metadata.items())
               for item in [MTDTA_FIRST, MTDTA_BURST]):

            descent_starts_at = prf.info.metadata[MTDTA_BURST] - prf.info.metadata[MTDTA_FIRST]
            if descent_starts_at.total_seconds() <= 0:
                logger.error('No ascent data (%s)', prf.info.src)
            is_descent = prf.data.index.get_level_values(PRF_TDT) > descent_starts_at

        else:
            max_alt_id = prf.data.index.get_level_values(PRF_ALT).argmax()
            is_descent = prf.data.index.get_level_values(PRF_IDX) > max_alt_id
            if is_descent.any():
                logger.warning('No burst time found in metadata (%s)',
                               prf.info.src)
                logger.info(
                    'Points after max alt %.1f [%s] @ %.1f [s] will be flagged as "%s". (%s)',
                    prf.data.index[max_alt_id][1],
                    prfs.var_info[PRF_ALT]['prm_unit'],
                    prf.data.index[max_alt_id][2].total_seconds(),
                    FLG_DESCENT, prf.info.src)

        # Actually set the flags
        prf.set_flg(FLG_DESCENT, True, index=is_descent)
        prf.set_flg(FLG_PRELAUNCH, True, index=is_prelaunch)
        prf.set_flg(FLG_ASCENT, True, index=~is_descent * ~is_prelaunch)

        # Sanity check
        assert not (prf.has_flg(FLG_DESCENT) * prf.has_flg(FLG_ASCENT)).any()
        assert not (prf.has_flg(FLG_PRELAUNCH) * prf.has_flg(FLG_ASCENT)).any()
        assert not (prf.has_flg(FLG_DESCENT) * prf.has_flg(FLG_PRELAUNCH)).any()

    return prfs


@log_func_call(logger, time_it=False)
def cleanup_steps(prfs, resampling_freq, interp_dist, crop_descent, crop_flgs,
                  timeofday=None, fid=None):
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
        crop_flgs (list): any data flagged with any one of these flag names will be cropped out for
            good.
        timeofday (str): if set, will tag the Profile with this time of day. Defaults to None.
        fid (str): if set, will add the flight id to the profile metadata. Defaults to None.

    Note:
        Pre-launch data is always cropped.

    """

    # Flag pre-launch, ascent, and descent data
    prfs = flag_phases(prfs)

    # Crop any prelaunch data
    for (ind, prf) in enumerate(prfs):

        # Crop pre-flight points, if warranted
        if prf.has_flg(FLG_PRELAUNCH).any():
            logger.info('Cropping pre-launch datapoints (%s)', prf.info.src)
            prfs[ind].data = prf.data.loc[~prf.has_flg(FLG_PRELAUNCH)]
            # Update the metadata (fixes #295)
            # Note that we are here BEFORE the resampling takes place. This implies that the
            # new value of MTDTA_FIRST cannot be computed by counting the number of cropped
            # time steps. It must be set to have the same value as MTDTA_LAUNCH
            prfs[ind].info.add_metadata(MTDTA_FIRST, prf.info.metadata[MTDTA_LAUNCH])

    # Basic sanity check of the input
    if crop_flgs is None:
        crop_flgs = []
    # Fail early if I need to ...
    assert isinstance(crop_flgs, list)
    # Include the descent flag is warranted
    if crop_descent and FLG_DESCENT not in crop_flgs:
        crop_flgs += [FLG_DESCENT]
    # Crop the requested flag names
    for flg_name in crop_flgs:
        for (ind, prf) in enumerate(prfs):
            prfs[ind].data = prf.data.loc[~prf.has_flg(flg_name)]

    # Add the TimeOfDay tag, if warranted
    if timeofday is not None:
        for (ind, prf) in enumerate(prfs):
            if not prf.has_tag(timeofday):
                prf.info.add_tags(timeofday)
                logger.info('Adding missing TimeOfDay tag (%s)', prf.info.src)

    # Add the fid, if warranted
    if fid is not None:
        for prf in prfs:
            prf.info.add_metadata('fid', fid)

    # Resample the profiles as required
    prfs.resample(freq=resampling_freq, interp_dist=interp_dist, inplace=True,
                  chunk_size=dynamic.CHUNK_SIZE, n_cpus=dynamic.N_CPUS)

    # Save back to the DB
    prfs.save_to_db(
        add_tags=[TAG_CLN, dynamic.CURRENT_STEP_ID],
        rm_tags=dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID)
        )


@for_each_var(incl_latlon=True)
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
    (fid, eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    filt = tools.get_query_filter(tags_in=tags + [eid, rid], tags_out=None)

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
                              ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                              uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                              ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                              inplace=True)

        logger.info('Loaded %i GDP profiles from the DB.', len(gdp_prfs))

        # Deal with the faulty RS41 gph_uc_tcor values (NaNs when they should not be, see #205)
        if dynamic.CURRENT_VAR == 'gph' and fix_gph_uct is not None:
            # Safety check
            if not isinstance(fix_gph_uct, list):
                raise DvasRecipesError(f'fix_gph_uct should be a list, not: {fix_gph_uct}')

            # Start looping through the profiles, to identify the ones I need.
            for gdp in gdp_prfs:
                if ' '.join(gdp.info.mid) in fix_gph_uct:

                    # What are the bug conditions ?
                    cond1 = gdp.data.loc[:, 'uct'].isna()
                    cond2 = ~gdp.data.loc[:, 'val'].isna()

                    if (n_bad := (cond1 & cond2).sum()) > 0:  # True = 1
                        logger.info('Fixing %i bad gph_uct values for mid: %s',
                                    n_bad, gdp.info.mid[0])
                        # Fix the bug
                        gdp.data.loc[cond1 & cond2, 'uct'] = gdp.data.loc[:, 'uct'].max(skipna=True)

                        # Flag it so we can find these bad points later on if needed
                        gdp.set_flg(FLG_WASINVALID, True,
                                    index=gdp.data.index[cond1 & cond2].values)

        # Check that whenever I have a value, I have a valid error, and vice-versa
        # Check also cases where the uncertainty is equal to 0
        # Both these checks are implemented following #244 and #260
        val_vs_uc = gdp_prfs.get_prms(['val', 'uc_tot'])
        for (gdp_ind, gdp) in enumerate(gdp_prfs):
            ok = val_vs_uc.loc[:, gdp_ind][PRF_VAL].isna() == \
                 val_vs_uc.loc[:, gdp_ind]['uc_tot'].isna()

            nok = val_vs_uc.loc[:, gdp_ind]['uc_tot'] == 0

            if dynamic.CURRENT_VAR not in ['lat', 'lon'] and (any(~ok) or any(nok)):
                logger.info('%s: %i/%i val vs uc_tot "NaN" mismatch for %s, flagged as "%s".',
                            '+'.join(gdp.info.mid), len(ok[~ok]), len(ok),
                            dynamic.CURRENT_VAR, FLG_ISINVALID)
                logger.info('%s: %i/%i val vs uc_tot "0" mismatch for %s, flagged as "%s".',
                            '+'.join(gdp.info.mid), len(nok[nok]), len(nok),
                            dynamic.CURRENT_VAR, FLG_ISINVALID)

                gdp_prfs[gdp_ind].set_flg(FLG_ISINVALID, True, index=ok.index[~ok].values)
                gdp_prfs[gdp_ind].set_flg(FLG_ISINVALID, True, index=nok.index[nok].values)

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

                        msg = f'Tropopause: GRUAN-> {val} [m] vs {dvas_trop[1]:.2f} [m] <-dvas ' +\
                            f'({gdp_prf.info.src})'

                        if abs(float(val)-dvas_trop[1]) <= 20:
                            logger.info(msg)
                        else:
                            logger.error(msg)

                    case _:
                        logger.error('Unknown GRUAN tropopause format: %s',
                                     gdp_prf.info.metadata['gruan_tropopause'])

        # Now launch more generic cleanup steps
        cleanup_steps(gdp_prfs, **args, timeofday=timeofday, fid=fid)

    # Process the non-GDPs, if any
    if not db_view[this_flight].is_gdp.all() and dynamic.CURRENT_VAR not in ['lat', 'lon']:
        logger.info('Cleaning non-GDP profiles for flight %s and variable %s',
                    dynamic.CURRENT_FLIGHT,
                    dynamic.CURRENT_VAR)

        # Extract the data from the db
        rs_prfs = MultiRSProfile()
        rs_prfs.load_from_db(f'and_({filt}, not_(tags("{TAG_GDP}")))',
                             dynamic.CURRENT_VAR, dynamic.INDEXES[PRF_TDT],
                             alt_abbr=dynamic.INDEXES[PRF_ALT])

        logger.info('Loaded %i RS profiles from the DB.', len(rs_prfs))

        cleanup_steps(rs_prfs, **args, timeofday=timeofday, fid=fid)
