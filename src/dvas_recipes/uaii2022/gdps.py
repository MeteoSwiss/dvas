"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level GDP recipes for the UAII2022 campaign
"""

# Import from python
import logging
import numpy as np
import pandas as pd

# Import from dvas
from dvas.data.data import MultiRSProfile, MultiGDPProfile
from dvas.tools.gdps import stats as dtgs
from dvas.tools.gdps import gdps as dtgg
import dvas.plots.gdps as dpg
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, FLG_INCOMPATIBLE_NAME
from dvas.hardcoded import PRF_REF_VAL_NAME, PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME
from dvas.hardcoded import PRF_REF_UCU_NAME, TAG_CWS_NAME, TAG_GDP_NAME
from dvas.errors import DBIOError

# Import from dvas_recipes
from ..errors import DvasRecipesError
from ..recipe import for_each_flight, for_each_var
from .. import dynamic
from ..utils import fn_suffix

# Setup local logger
logger = logging.getLogger(__name__)


@for_each_var
@for_each_flight
def build_cws(tags='sync', m_vals=None, strategy='all-or-none', alpha=0.0027):
    """ Highest-level recipe function responsible for assembling the combined working standard for
    a specific RS flight.

    This function directly builds the profiles and uploads them to the db with the 'cws' tag.

    Args:
        tags (str|list of str, optional): tag name(s) for the search query into the database.
            Defaults to 'sync'.
        m_vals (int|list of int, optional): list of m-values used for identifiying incompatible and
            valid regions between GDPs. Any negative value will be ignored when computing the cws.
            Defaults to None=[1, '-2'].
        strategy (str, optional): name of GDP combination strategy (for deciding which levels/
            measurements are valid or not). Defaults to 'all-or-none'. These ared defined in
            `dvas.tools.gdps.stats.get_validities()`.
        alpha (float, optional): The significance level for the KS test. Defaults to 0.27%.
            See dvas.tools.gdps.stats.gdp_incompatibilities() for details.

    TODO:
        Give the user the possibility to tag the 'cws' with a custom one ?

    """

    # Deal with the search tags
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        raise DvasRecipesError('Ouch ! tags should be of type str|list. not: {}'.format(type(tags)))

    # Deal with the m_vals if warranted
    if m_vals is None:
        m_vals = [1, -2]
    if not isinstance(m_vals, list):
        raise DvasRecipesError(f'Ouch ! m_vals should be a list of int, not: {m_vals}')

    # Get the event id and rig id
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    gdp_filt = "and_(tags('gdp'), tags('e:{}'), tags('r:{}'), {})".format(
        eid, rid, "tags('" + "'), tags('".join(tags) + "')")
    cws_filt = "and_(tags('cws'), tags('e:{}'), tags('r:{}'))".format(eid, rid)

    # Load the GDP profiles
    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(gdp_filt, dynamic.CURRENT_VAR,
                          tdt_abbr=dynamic.INDEXES[PRF_REF_TDT_NAME],
                          alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME],
                          ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                          ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                          uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                          ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                          inplace=True)

    # First things first, let's check if we have already computed the 'tdt' profile of the CWS
    # during a previous pass. If not, create it now. We treat the case of the tdt sepearately,
    # since we compute it as a normal (unweighted) mean.
    try:
        logger.info('Looking for a pre-computed CWS time profile ...')
        _ = MultiRSProfile().load_from_db(cws_filt, 'time', 'time')

    except DBIOError:
        logger.info('No pre-existing CWS time profile found. Computing it now.')
        # Very well, let us compute the time array of the combined working standard. Since there is
        # no uncertainty associated to it, we shall simply take it as the arithmetic mean of the
        # individual GDP times.
        # First, load the time data

        # Let us now create a high-resolution CWS for these synchronized GDPs
        tdt_cws = dtgg.combine(gdp_prfs, binning=1, method='mean',
                               mask_flgs=None, chunk_size=dynamic.CHUNK_SIZE,
                               n_cpus=dynamic.N_CPUS)

        # TODO: doing a mean of the tdt from each radiosonde is a problem, because if they are not
        # synced, we may end up with jumps backwards, or weird stuff.
        # What we need is a proper function to identify the start time of the flight, and use this
        # to build the time delta array of the CWS.
        # For now, just assume step 0 is the launch (not always TRUE!)
        tdt_cws[0].data.reset_index(level='tdt', drop=True, inplace=True)
        tdt_cws[0].data['tdt'] = pd.Series(np.arange(0, len(tdt_cws[0].data), 1)*1e9,
                                           dtype='timedelta64[ns]').values
        tdt_cws[0].data.set_index('tdt', inplace=True, drop=True, append=True)

        # And then save the tdt array to the database ... tdt and only tdt !
        tdt_cws.save_to_db(add_tags=['cws'], rm_tags=['gdp'], prms=[PRF_REF_TDT_NAME])

    # Before combining the GDPs with each other, let us assess their consistency.
    # The idea here is to flag any inconsistent measurement, so that they can be ignored during
    # the combination process.
    logger.info('Identifying incompatibilities between GDPs for variable: %s', dynamic.CURRENT_VAR)
    incompat = dtgs.gdp_incompatibilities(gdp_prfs, alpha=alpha,
                                          m_vals=[np.abs(item) for item in m_vals],
                                          do_plot=True,
                                          n_cpus=dynamic.N_CPUS,
                                          chunk_size=dynamic.CHUNK_SIZE,
                                          fn_prefix=dynamic.CURRENT_STEP_ID,
                                          fn_suffix=fn_suffix(eid=eid, rid=rid, tags=tags,
                                                              var=dynamic.CURRENT_VAR))

    # Next, we derive "validities" given a specific strategy to assess the different GDP pair
    # incompatibilities ...
    # Note how the m_vals used for the combination can differ from the ones used to check the
    # incompatibilities. This is intended to let people experiment a bit without affecting the final
    # CWS.
    valids = dtgs.gdp_validities(incompat,
                                 m_vals=[item for item in m_vals if item > 0],
                                 strategy=strategy)

    # ... and set them using the dvas.hardcoded.FLG_INCOMPATIBLE_NAME flag
    for gdp_prf in gdp_prfs:
        gdp_prf.set_flg(FLG_INCOMPATIBLE_NAME, True,
                        index=valids[~valids[str(gdp_prf.info.oid)]].index)

    # Let us now create a high-resolution CWS for these synchronized GDPs
    cws = dtgg.combine(gdp_prfs, binning=1, method='weighted mean',
                       mask_flgs=FLG_INCOMPATIBLE_NAME,
                       chunk_size=dynamic.CHUNK_SIZE, n_cpus=dynamic.N_CPUS)

    # We can now inspect the result visually
    dpg.gdps_vs_cws(gdp_prfs, cws, index_name='_idx', show=None, fn_prefix=dynamic.CURRENT_STEP_ID,
                    fn_suffix=fn_suffix(eid=eid, rid=rid, tags=tags, var=dynamic.CURRENT_VAR))

    # Save the CWS to the database
    # Here, I only save the information associated to the variable, i.e. the value and its errors.
    # I do not save the alt column, which is a variable itself and should be derived as such using a
    # weighted mean. I also do not save the tdt column, which should be assembled from a simple mean
    cws.save_to_db(add_tags=[TAG_CWS_NAME], rm_tags=[TAG_GDP_NAME],
                   prms=[PRF_REF_VAL_NAME,
                         PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME])

    #TODO: if the variable is the altitude, save it under a different variable name ...
