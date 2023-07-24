"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level delta recipes for the UAII2022 campaign
"""

# Import from python
import logging

# Import from dvas
from dvas.logger import log_func_call
from dvas.data.data import MultiProfile, MultiCWSProfile
from dvas.tools.dtas import dtas as dtdd
from dvas.plots import dtas as dpd
from dvas.hardcoded import PRF_TDT, PRF_ALT
from dvas.hardcoded import TAG_DTA, TAG_GDP, TAG_CWS

# Import from dvas_recipes
from ..errors import DvasRecipesError
from ..recipe import for_each_flight, for_each_var
from .. import dynamic
from . import tools
from .. import utils as dru

# Setup local logger
logger = logging.getLogger(__name__)


@for_each_var(incl_wvec=True)
@for_each_flight
@log_func_call(logger)
def compute_deltas(prf_start_with_tags, cws_start_with_tags, do_gdps=False, do_nongdps=True,
                   save_to_db=False):
    """ Highest-level recipe function responsible for computing differences between profiles under
    test and appropriate combined working standards.

    Args:
        prf_start_with_tags (str|list of str): tag name(s) for the search query into the database.
        cws_start_with_tags (str|list of str): cws tag name(s) for the search into the database.
        do_gdps (bool, optional): if True, will also compute the deltas for GDP profiles
            (ignoring the GDP uncertaintes entirely). Defaults to False.
        do_nongdps (bool): if True, will also compute the deltas for the non-GDP profiles.
            Default to True.
        save_to_db (bool optional): if True, the deltas will be saved to the DB with the 'delta'
            tag.

    This function directly builds the DeltaProfile instances and uploads them to the db with the
    'delta' tag.

    Note:
        This function is designed to work for one specific flight at a time. It can then be looped
        over using the @for_each_flight decorator to process the entire campaign.


    """

    # Cleanup the tags
    prf_tags = dru.format_tags(prf_start_with_tags)
    cws_tags = dru.format_tags(cws_start_with_tags)

    # Get the event id and rig id
    (fid, eid, rid) = dynamic.CURRENT_FLIGHT

    # What tags should I exclude from the search ?
    if not do_gdps and not do_nongdps:
        raise DvasRecipesError('incl_gdps and incl_nongdps cannot both be False.')

    if do_gdps:
        tags_out = [TAG_CWS]
    else:
        tags_out = [TAG_GDP, TAG_CWS]

    if do_nongdps:
        tags_in = prf_tags + [eid, rid]
    else:
        tags_in = prf_tags + [eid, rid, TAG_GDP]

    # What search query will let me access the data I need ?
    prf_filt = tools.get_query_filter(tags_in=tags_in,
                                      tags_out=tags_out)
    cws_filt = tools.get_query_filter(
        tags_in=cws_tags + [eid, rid, TAG_CWS], tags_out=None)

    # Deal with the CWS variables
    if dynamic.CURRENT_VAR in dru.cws_vars():
        # Load the non GDP profiles as Profiles (and not RSProfiles) since we're about to drop the
        # time axis anyway.
        prfs = MultiProfile()
        prfs.load_from_db(prf_filt, dynamic.CURRENT_VAR,
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          inplace=True)

        # Load the CWS
        cws_prfs = MultiCWSProfile()
        cws_prfs.load_from_db(cws_filt, dynamic.CURRENT_VAR,
                              tdt_abbr=dynamic.INDEXES[PRF_TDT],
                              alt_abbr=dynamic.INDEXES[PRF_ALT],
                              ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                              uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                              ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                              inplace=True)

        dir_prfs = None
        dir_cws_prfs = None

        # Safety check for the CWS
        if len(cws_prfs) != 1:
            raise DvasRecipesError(f'I need 1 CWS, but I got {len(cws_prfs)} instead.')

    elif dynamic.CURRENT_VAR == 'wvec':
        # Let's deal with the special case of the wind (horizontal) vector
        # This is to be consistent with OSCAR, and requires dealing with both the wind speed
        # and the wind direction.

        # WARNING: hardcoded variable name ... !!!
        prfs = MultiProfile()
        prfs.load_from_db(prf_filt, 'wspeed',
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          inplace=True)

        # Load the CWS
        cws_prfs = MultiCWSProfile()
        cws_prfs.load_from_db(cws_filt, 'wspeed',
                              tdt_abbr=dynamic.INDEXES[PRF_TDT],
                              alt_abbr=dynamic.INDEXES[PRF_ALT],
                              ucs_abbr=dynamic.ALL_VARS['wspeed']['ucs'],
                              uct_abbr=dynamic.ALL_VARS['wspeed']['uct'],
                              ucu_abbr=dynamic.ALL_VARS['wspeed']['ucu'],
                              inplace=True)

        # Safety check for the CWS
        if len(cws_prfs) != 1:
            raise DvasRecipesError(f'I need 1 CWS, but I got {len(cws_prfs)} instead.')

        dir_prfs = MultiProfile()
        dir_prfs.load_from_db(prf_filt, 'wdir',
                              alt_abbr=dynamic.INDEXES[PRF_ALT],
                              inplace=True)

        # Load the CWS
        dir_cws_prfs = MultiCWSProfile()
        dir_cws_prfs.load_from_db(cws_filt, 'wdir',
                                  tdt_abbr=dynamic.INDEXES[PRF_TDT],
                                  alt_abbr=dynamic.INDEXES[PRF_ALT],
                                  ucs_abbr=dynamic.ALL_VARS['wdir']['ucs'],
                                  uct_abbr=dynamic.ALL_VARS['wdir']['uct'],
                                  ucu_abbr=dynamic.ALL_VARS['wdir']['ucu'],
                                  inplace=True)

        # Safety check for the CWS
        if len(dir_cws_prfs) != 1:
            raise DvasRecipesError(f'I need 1 wdir CWS, but I got {len(dir_cws_prfs)} instead.')
    else:
        raise DvasRecipesError(f'Unknown variable: {dynamic.CURRENT_VAR}')

    # Compute the Delta Profiles
    dta_prfs = dtdd.compute(prfs, cws_prfs, circular=dynamic.CURRENT_VAR == 'wdir',
                            dir_prfs=dir_prfs, dir_cwss=dir_cws_prfs)

    # Save the Delta profiles to the database.
    if save_to_db:
        logger.info('Saving delta profiles to the DB.')
        dta_prfs.save_to_db(add_tags=[TAG_DTA, dynamic.CURRENT_STEP_ID],
                            rm_tags=dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID))

    # Let us now also plot these deltas
    fn_suf = dru.fn_suffix(fid=fid, eid=eid, rid=rid, tags=prf_tags, var=dynamic.CURRENT_VAR)

    if do_gdps and do_nongdps:
        fn_suf += ''
    elif do_gdps:
        fn_suf += '_gdps'
    elif do_nongdps:
        fn_suf += '_non-gdps'

    dpd.dtas(dta_prfs, k_lvl=1, label='mid', show=False,
             fn_prefix=dynamic.CURRENT_STEP_ID, fn_suffix=fn_suf)
