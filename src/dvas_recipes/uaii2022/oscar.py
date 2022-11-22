"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level OSCAR-related routines for the UAII 2022 recipe
"""

# Import from python
import logging

# Import from dvas
from dvas.logger import log_func_call
from dvas.data.data import MultiDeltaProfile
from dvas.dvas import Database as DB
#from dvas.tools.dtas import dtas as dtdd
#from dvas.plots import dtas as dpd
from dvas.hardcoded import PRF_TDT, PRF_ALT
from dvas.hardcoded import TAG_DTA, TAG_GDP, TAG_CWS

# Import from dvas_recipes
from ..errors import DvasRecipesError
from ..recipe import for_each_var
from .. import dynamic
from . import tools
from .. import utils as dru

# Setup local logger
logger = logging.getLogger(__name__)


@for_each_var
@log_func_call(logger)
def compute_oscar(start_with_tags, mids=None):
    """ Highest-level recipe function responsible for assembling OSCAR profiles

    Args:
        start_with_tags (str|list of str): tag name(s) for the search query into the database.
        mids (list, optional): list of 'mid' to process. Defaults to None = all

    """

    # Cleanup the tags
    prf_tags = dru.format_tags(start_with_tags)

    # Very well, let us first extract the 'mid', if they have not been provided
    db_view = DB.extract_global_view()
    if mids is None:
        mids = db_view.mid.unique().tolist()

    # Basic sanity check of mid
    if not isinstance(mids, list):
        raise DvasRecipesError(f'Ouch ! I need a list of mids, not: {mids}')

    # Very well, let's now loop through these, and compute the OSCAR profiles
    for mid in mids:

        # Second sanity check - make sure the mid is in the DB
        if mid not in db_view.mid.unique().tolist():
            raise DvasRecipesError(f'mid unknown: {mid}')
        else:
            logger.info('Processing %s ...', mid)

        # What search query will let me access the data I need ?
        prf_filt = tools.get_query_filter(tags_in=prf_tags + [TAG_DTA],
                                          tags_out=dru.rsid_tags(pop=prf_tags) + [TAG_CWS],
                                          mids=mid)

        # Load the delta profiles as Profiles (and not RSProfiles) since we're about to drop the
        prfs = MultiDeltaProfile()
        prfs.load_from_db(prf_filt, dynamic.CURRENT_VAR,
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          inplace=True)
        logger.info('Found %i delta_profiles', len(prfs))

        import pdb
        pdb.set_trace()

    """
    # Compute the Delta Profiles
    dta_prfs = dtdd.compute(prfs, cws_prfs, circular=dynamic.CURRENT_VAR == 'wdir')

    # Save the Delta profiles to the database.
    # WARNING: I will keep the GDP tag, even if the resulting delta profile is not fully correct
    # in terms of error propagation. This is just to still be able to distinguish between those
    # GDP and non-GDP profiles down the line.
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
    """