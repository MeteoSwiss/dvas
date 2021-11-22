"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level delta recipes for the UAII22 campaign
"""

# Import from python
import numpy as np
import pandas as pd

# Import from dvas
from dvas.data.data import MultiProfile, MultiCWSProfile
from dvas.tools.dtas import dtas as dtdd
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, FLG_INCOMPATIBLE_NAME
from dvas.hardcoded import PRF_REF_VAL_NAME, PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME
from dvas.hardcoded import PRF_REF_UCU_NAME, TAG_DTA_NAME, TAG_GDP_NAME, TAG_CWS_NAME
from dvas.errors import DBIOError

# Import from dvas_recipes
from ..errors import DvasRecipesError
from ..recipe import for_each_flight, for_each_var
from .. import dynamic
from ..utils import fn_suffix

@for_each_var
@for_each_flight
def compute_deltas(tags='sync', mids='all'):
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
        mids (str|list, optional): list of model ids to process. Defaults to 'all'.
            CURRENTLY HAS NO EFFECT !

    """

    # Deal with the search tags
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        raise DvasRecipesError('Ouch ! tags should be of type str|list. not: {}'.format(type(tags)))

    # Deal with the mids
    # TODO: actually allow users to only process specific models. We need to implement #168 first.

    # Get the event id and rig id
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    nongdp_filt = "and_(not_(tags('gdp')), tags('e:{}'), tags('r:{}'), {})".format(eid, rid,
        "tags('" + "'), tags('".join(tags) + "')")
    cws_filt = "and_(tags('cws'), tags('e:{}'), tags('r:{}'))".format(eid, rid)

    # Load the non GDP profiles as Profiles (and not RSProfiles) since we're about to drop the
    # time axis anyway.
    nongdp_prfs = MultiProfile()
    nongdp_prfs.load_from_db(nongdp_filt, dynamic.CURRENT_VAR,
                             #tdt_abbr=dynamic.INDEXES[PRF_REF_TDT_NAME],
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
        raise DvasRecipesError('Ouch ! I need 1 CWS, but I got %i instead.', len(cws_prfs))

    # Compute the Delta Profiles
    dta_prfs = dtdd.compute(nongdp_prfs, cws_prfs)

    # TODO: inspect the result visually

    # Save the Delta profiles to the database
    # Here, I only save the information associated to the variable, i.e. the value and its errors.
    # I do not save the alt column, which is a variable itself and should be derived as such using a
    # weighted mean. I also do not save the tdt column, which should be assembled from a simple mean
    dta_prfs.save_to_db(add_tags=[TAG_DTA_NAME], rm_tags=[TAG_GDP_NAME, TAG_CWS_NAME],
                        prms=[PRF_REF_VAL_NAME, PRF_REF_UCR_NAME, PRF_REF_UCS_NAME,
                              PRF_REF_UCT_NAME, PRF_REF_UCU_NAME])

    # TODO: export the altitude separately, into a dedicated variable of the DB.
