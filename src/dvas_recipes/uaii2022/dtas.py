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
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME
from dvas.hardcoded import TAG_DTA_NAME, TAG_GDP_NAME, TAG_CWS_NAME

# Import from dvas_recipes
from ..errors import DvasRecipesError
from ..recipe import for_each_flight, for_each_var
from .. import dynamic
from . import tools
from ..utils import fn_suffix

# Setup local logger
logger = logging.getLogger(__name__)


@for_each_var
@for_each_flight
@log_func_call(logger)
def compute_deltas(tags='sync', incl_gdps=False, save_to_db=False):
    """ Highest-level recipe function responsible for compute differences between profiles under
    test and appropriate combined working standards.

    This function directly builds the DeltaProfile instances and uploads them to the db with the
    'delta' tag.

    Note:
        This function is designed to work for one specific flight at a time. It can then be looped
        over using the @for_each_flight decorator to process the entire campaign.

    Args:
        tags (str|list of str, optional): tag name(s) for the search query into the database.
            Defaults to 'sync'.
        incl_gdps (bool, optional): if True, will also compute the deltas for GDP profiles
            (ignoring the GDP uncertaintes entirely). Defaults to False.
        save_to_db (bool optional): if True, the deltas will be saved to the DB with the 'delta'
            tag.

    """

    # Deal with the search tags
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        raise DvasRecipesError('Ouch ! tags should be of type str|list. not: {}'.format(type(tags)))

    # Get the event id and rig id
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What tags should I exclude from the search ?
    if incl_gdps:
        tags_out = [TAG_CWS_NAME, TAG_DTA_NAME]
    else:
        tags_out = [TAG_GDP_NAME, TAG_CWS_NAME, TAG_DTA_NAME]

    # What search query will let me access the data I need ?
    nongdp_filt = tools.get_query_filter(tags_in=tags+[eid, rid], tags_out=tags_out)
    cws_filt = tools.get_query_filter(tags_in=tags+[eid, rid, TAG_CWS_NAME], tags_out=None)

    # Load the non GDP profiles as Profiles (and not RSProfiles) since we're about to drop the
    # time axis anyway.
    nongdp_prfs = MultiProfile()
    nongdp_prfs.load_from_db(nongdp_filt, dynamic.CURRENT_VAR,
                             # tdt_abbr=dynamic.INDEXES[PRF_REF_TDT_NAME],
                             alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME],
                             inplace=True)

    # Load the CWS
    cws_prfs = MultiCWSProfile()
    cws_prfs.load_from_db(cws_filt, dynamic.CURRENT_VAR,
                          tdt_abbr=dynamic.INDEXES[PRF_REF_TDT_NAME],
                          alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME],
                          ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                          ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                          uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                          ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                          inplace=True)

    # Safety check for the CWS
    if len(cws_prfs) != 1:
        raise DvasRecipesError(f'Ouch ! I need 1 CWS, but I got {len(cws_prfs)} instead.')

    # Compute the Delta Profiles
    dta_prfs = dtdd.compute(nongdp_prfs, cws_prfs)

    # Save the Delta profiles to the database.
    # WARNING: I will keep the GDP tag, even if the resulting delta profile is not fully correct
    # in terms of error propagation. This is just to still be able to distinguish between those
    # GDP and non-GDP profiles down the line.
    if save_to_db:
        logger.info('Saving delta profiles to the DB.')
        dta_prfs.save_to_db(add_tags=[TAG_DTA_NAME], rm_tags=[])

    # Let us now also plot these deltas
    fn_suf = fn_suffix(eid=eid, rid=rid, tags=tags, var=dynamic.CURRENT_VAR)

    if incl_gdps:
        fn_suf += '_with-gdps'

    dpd.dtas(dta_prfs, k_lvl=1, label='mid', show=True,
             fn_prefix=dynamic.CURRENT_STEP_ID, fn_suffix=fn_suf)
