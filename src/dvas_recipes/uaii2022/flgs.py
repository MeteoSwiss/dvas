"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level flagging recipes for the UAII2022 campaign
"""

# Import from Python
import logging

# Import from dvas
from dvas.logger import log_func_call
from dvas.hardcoded import TAG_CWS, TAG_GDP, TAG_DTA, FLG_HASCWS
from dvas.hardcoded import PRF_TDT, PRF_ALT, PRF_VAL
from dvas.hardcoded import MTDTA_TROPOPAUSE
from dvas.data.data import MultiRSProfile, MultiGDPProfile, MultiCWSProfile
from dvas.errors import DBIOError

# Import from dvas_recipes
from .. import dynamic
from ..errors import DvasRecipesError
from .. import utils as dru
from ..recipe import for_each_flight
from . import tools

# Setup local logger
logger = logging.getLogger(__name__)


@for_each_flight
@log_func_call(logger, time_it=False)
def set_zone_flags(prf_tags=None, cws_tags=None, temp_var='temp'):
    """ Flag the different regions of interest in the Profiles, e.g. "descent", "FT", "UTLS", ...

    Args:
        prf_tags (str, list): which tags to use to identify Profiles in the DB.
            Defaults to None.
        cws_tags (str, list): which tags to use to identify CWS in the DB.
            Defaults to None.
        temp_var (str, optional): name of the temperature variable, to derive the troposphere
            altitude. Defaults to 'temp'.

    """

    # Cleanup the tags
    prf_tags = dru.format_tags(prf_tags)
    cws_tags = dru.format_tags(cws_tags)

    # Get the event id and rig id
    (_, eid, rid) = dynamic.CURRENT_FLIGHT

    # Define the DB query filters that will get me what I want
    cws_filt = tools.get_query_filter(
        tags_in=cws_tags+[eid, rid, TAG_CWS],
        tags_out=dru.rsid_tags(pop=cws_tags) + [TAG_GDP, TAG_DTA])

    gdp_filt = tools.get_query_filter(
        tags_in=prf_tags+[eid, rid, TAG_GDP],
        tags_out=dru.rsid_tags(pop=prf_tags) + [TAG_CWS, TAG_DTA])

    nongdp_filt = tools.get_query_filter(
        tags_in=prf_tags+[eid, rid],
        tags_out=dru.rsid_tags(pop=prf_tags) + [TAG_GDP, TAG_CWS, TAG_DTA])

    # TODO: get the PBL limit info somehow - fixed value for site, or flight-per-flight basis ?

    # Step 1: query the temp CWS, so that we can derive the tropopause altitude
    if temp_var not in dynamic.ALL_VARS.keys():
        raise DvasRecipesError(f'Temperature variable "{temp_var}" does not exist ?!')

    # Load the CWS as a simple RS profile, since I do not need the uncertainties to derive the
    # tropopause altitude.
    cws_prfs = MultiRSProfile()
    cws_prfs.load_from_db(cws_filt, temp_var,
                          tdt_abbr=dynamic.INDEXES[PRF_TDT],
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          inplace=True)

    if (_ := len(cws_prfs)) != 1:
        raise DvasRecipesError(f'Found {_} CWS profiles for {temp_var}, but expected exactly 1.')

    _, pbl_alt, pbl_tdt = tools.find_tropopause(cws_prfs[0])
    logger.info('Tropopause from CWS: %.2f [m] @ %s', pbl_alt, pbl_tdt)

    # For all variables, fetch the CWS to identify the valid regions
    for var_name in dynamic.ALL_VARS:
        logger.info('Flagging zones for variable %s', var_name)

        cws_prfs = MultiCWSProfile()
        cws_prfs.load_from_db(cws_filt, var_name,
                              tdt_abbr=dynamic.INDEXES[PRF_TDT],
                              alt_abbr=dynamic.INDEXES[PRF_ALT],
                              ucr_abbr=dynamic.ALL_VARS[var_name]['ucr'],
                              ucs_abbr=dynamic.ALL_VARS[var_name]['ucs'],
                              uct_abbr=dynamic.ALL_VARS[var_name]['uct'],
                              ucu_abbr=dynamic.ALL_VARS[var_name]['ucu'],
                              inplace=True)

        if len(cws_prfs) != 1:
            raise DvasRecipesError('Got more than 1 CWS ... ?!')

        # Where do I have a valid CWS ?
        cws_cond = cws_prfs[0].data[PRF_VAL].notna().values

        gdp_prfs = MultiGDPProfile()
        try:
            gdp_prfs.load_from_db(gdp_filt, var_name,
                                  tdt_abbr=dynamic.INDEXES[PRF_TDT],
                                  alt_abbr=dynamic.INDEXES[PRF_ALT],
                                  ucr_abbr=dynamic.ALL_VARS[var_name]['ucr'],
                                  ucs_abbr=dynamic.ALL_VARS[var_name]['ucs'],
                                  uct_abbr=dynamic.ALL_VARS[var_name]['uct'],
                                  ucu_abbr=dynamic.ALL_VARS[var_name]['ucu'],
                                  inplace=True)
        except DBIOError:
            logger.debug('No GDP profile found.')
            gdp_prfs = None

        nongdp_prfs = MultiRSProfile()
        try:
            nongdp_prfs.load_from_db(nongdp_filt, var_name,
                                     tdt_abbr=dynamic.INDEXES[PRF_TDT],
                                     alt_abbr=dynamic.INDEXES[PRF_ALT],
                                     inplace=True)
        except DBIOError:
            logger.debug('No non-GDP profile found.')
            nongdp_prfs = None

        # Apply the PBL, FT, UTLS, HAS_CWS flags to every profile.
        for prfs in [cws_prfs, gdp_prfs, nongdp_prfs]:

            if prfs is None:
                continue

            for (prf_ind, prf) in enumerate(prfs):

                # Apply the valid CWS flag
                prfs[prf_ind].set_flg(FLG_HASCWS, True, index=cws_cond)

                # Tropopause, Troposphere & Stratosphere
                prfs[prf_ind].info.add_metadata(MTDTA_TROPOPAUSE, pbl_alt)  # TODO: deal with units
                t_cond = prf.data.index.get_level_values(PRF_ALT).values < pbl_alt
                prfs[prf_ind].set_flg('troposphere', True, index=t_cond)
                s_cond = prf.data.index.get_level_values(PRF_ALT).values > pbl_alt
                prfs[prf_ind].set_flg('stratosphere', True, index=s_cond)

            # Save it all back into the DB
            prfs.save_to_db(add_tags=[dynamic.CURRENT_STEP_ID],
                            rm_tags=dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID))
