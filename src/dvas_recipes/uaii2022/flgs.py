"""
Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level flagging recipes for the UAII2022 campaign
"""

# Import from Python
import logging
import numpy as np

# Import from dvas
from dvas.logger import log_func_call
from dvas.hardcoded import TAG_CWS, TAG_GDP, FLG_HASCWS
from dvas.hardcoded import PRF_TDT, PRF_ALT, PRF_VAL
from dvas.hardcoded import FLG_PBL, FLG_FT, FLG_UTLS, FLG_MUS
from dvas.hardcoded import MTDTA_TROPOPAUSE, MTDTA_PBLH, MTDTA_UTLSMIN, MTDTA_UTLSMAX, MTDTA_MUSMIN
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
def set_zone_flags(prf_tags=None, cws_tags=None, temp_var='temp', set_pblh_at=None,
                   utls_lims=None, set_mus_above=None):
    """ Flag the different regions of interest in the Profiles, e.g. "descent", "FT", "UTLS", ...

    Args:
        prf_tags (str, list): which tags to use to identify Profiles in the DB.
            Defaults to None.
        cws_tags (str, list): which tags to use to identify CWS in the DB.
            Defaults to None.
        temp_var (str, optional): name of the temperature variable, to derive the troposphere
            altitude. Defaults to 'temp'.
        set_pblh_at (dict, optional): geopotential height of the PBLH
        utls_lims (dict, optional): dict with 'min' and 'max' keys specifying the UTLS upper and
            lower bounds, in geopotential meters. Defaults to None.
        set_mus_above (float, optional): geopotential height above which the MUS starts.

    """

    # Cleanup the tags
    prf_tags = dru.format_tags(prf_tags)
    cws_tags = dru.format_tags(cws_tags)

    # Get the event id and rig id
    (_, eid, rid) = dynamic.CURRENT_FLIGHT

    # Define the DB query filters that will get me what I want
    cws_filt = tools.get_query_filter(
        tags_in=cws_tags + [eid, rid, TAG_CWS],
        tags_out=None)

    gdp_filt = tools.get_query_filter(
        tags_in=prf_tags + [eid, rid, TAG_GDP],
        tags_out=None)

    nongdp_filt = tools.get_query_filter(
        tags_in=prf_tags + [eid, rid],
        tags_out=[TAG_GDP, TAG_CWS])

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

    _, tropopause_alt, tropopause_tdt = tools.find_tropopause(cws_prfs[0], algo='gruan')
    tropopause_unit = cws_prfs.var_info[PRF_ALT]['prm_unit']
    logger.info('Tropopause from CWS: %.2f [%s] @ %s',
                tropopause_alt, tropopause_unit, tropopause_tdt)

    # For all CWS variables, fetch the CWS to identify the valid regions
    for var_name in dru.cws_vars(incl_latlon=True):
        logger.info('Flagging zones for variable %s', var_name)

        cws_prfs = MultiCWSProfile()
        cws_prfs.load_from_db(cws_filt, var_name,
                              tdt_abbr=dynamic.INDEXES[PRF_TDT],
                              alt_abbr=dynamic.INDEXES[PRF_ALT],
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

                # Flag the different atmospheric regions ...
                # Tropopause
                prfs[prf_ind].info.add_metadata(
                    MTDTA_TROPOPAUSE, f"{tropopause_alt:.1f} {tropopause_unit}")

                # PBL
                if set_pblh_at is not None:
                    assert isinstance(set_pblh_at, dict), \
                        f'set_pbl_at should be dict, not: {type(set_pblh_at)}'

                    # Identify which tod tags are set
                    pbl_alt = [value for (key, value) in set_pblh_at.items()
                               if prf.has_tag(f'tod:{key}')]

                    if len(pbl_alt) > 1:
                        tods = [key for key in set_pblh_at.keys() if prf.has_tag(key)]
                        logger.error('Multiple tods for mid: %s -- %s', prf.info.mid, tods)

                    pbl_alt = np.unique(pbl_alt)
                    if len(pbl_alt) == 0:
                        raise DvasRecipesError(f'No PBL found ... invalid ToD ? {set_pblh_at}')
                    elif len(pbl_alt) > 1:
                        raise DvasRecipesError('Non-unique PBL ... duplicated ToD ?')
                    else:
                        pbl_alt = pbl_alt[0]

                    # Add the info the metadata
                    prfs[prf_ind].info.add_metadata(MTDTA_PBLH, f"{pbl_alt:.1f} m")

                    # Also apply the PBL flags
                    cond = cws_prfs[0].data.index.get_level_values(PRF_ALT).values < pbl_alt
                    prfs[prf_ind].set_flg(FLG_PBL, True, index=cond)

                    # With a PBL, I can also flag the free troposphere
                    cond = cws_prfs[0].data.index.get_level_values(PRF_ALT).values > pbl_alt
                    cond *= cws_prfs[0].data.index.get_level_values(PRF_ALT).values < tropopause_alt
                    prfs[prf_ind].set_flg(FLG_FT, True, index=cond)
                else:
                    logger.warning('No PBL info provided. No flags set for: %s, %s',
                                   FLG_PBL, FLG_FT)

                # UTLS
                if utls_lims is not None:
                    if not isinstance(utls_lims, dict):
                        raise DvasRecipesError(f'utls_lims is not dict: {type(utls_lims)}')
                    for key in ['min', 'max']:
                        if key not in utls_lims.keys():
                            raise DvasRecipesError(f'{key} key not found in utls_lims')

                    prfs[prf_ind].info.add_metadata(MTDTA_UTLSMIN,  f"{utls_lims['min']:.1f} m")
                    prfs[prf_ind].info.add_metadata(MTDTA_UTLSMAX,  f"{utls_lims['max']:.1f} m")

                    cond = cws_prfs[0].data.index.get_level_values(PRF_ALT).values > \
                        utls_lims['min']
                    cond *= cws_prfs[0].data.index.get_level_values(PRF_ALT).values < \
                        utls_lims['max']
                    prfs[prf_ind].set_flg(FLG_UTLS, True, index=cond)
                else:
                    logger.warning('No UTLS info provided. No flags set for: %s', FLG_UTLS)

                # MUS
                if set_mus_above is not None:
                    prfs[prf_ind].info.add_metadata(MTDTA_MUSMIN, f"{set_mus_above:.1f} m")
                    cond = cws_prfs[0].data.index.get_level_values(PRF_ALT) > set_mus_above
                    prfs[prf_ind].set_flg(FLG_MUS, True, index=cond)
                else:
                    logger.warning('No MUS info provided. No flags set for: %s', FLG_MUS)

            # Save it all back into the DB
            prfs.save_to_db(add_tags=[dynamic.CURRENT_STEP_ID],
                            rm_tags=dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID))
