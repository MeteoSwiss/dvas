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
from dvas.logger import log_func_call
from dvas.data.data import MultiRSProfile, MultiGDPProfile
from dvas.tools.gdps import stats as dtgs
from dvas.tools.gdps import gdps as dtgg
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, FLG_INCOMPATIBLE_NAME
from dvas.hardcoded import PRF_REF_VAL_NAME, PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME
from dvas.hardcoded import PRF_REF_UCU_NAME, TAG_CWS_NAME, TAG_GDP_NAME
from dvas.errors import DBIOError

# Import from dvas_recipes
from ..errors import DvasRecipesError
from ..recipe import for_each_flight, for_each_var
from .. import dynamic
from .. import utils as dru
from . import tools, plots

# Setup local logger
logger = logging.getLogger(__name__)


@for_each_var
@for_each_flight
@log_func_call(logger, time_it=True)
def build_cws(start_with_tags, m_vals=None, strategy='all-or-none',  method='weighted mean',
              alpha=0.0027, cws_alt_ref='gph', explore_covmats=True):
    """ Highest-level recipe function responsible for assembling the combined working standard for
    a specific RS flight.

    This function directly builds the profiles and uploads them to the db with the 'cws' tag.

    Args:
        start_with_tags (str|list of str): tag name(s) for the search query into the database.
        m_vals (int|list of int, optional): list of m-values used for identifiying incompatible and
            valid regions between GDPs. Any negative value will be ignored when computing the cws.
            Defaults to None=[1, '-2'].
        strategy (str, optional): name of GDP combination strategy (for deciding which levels/
            measurements are valid or not). Defaults to 'all-or-none'. These are defined in
            :py:func:`dvas.tools.gdps.stats.get_validities`.
        method (str, optional): combination method. Can be one of ['mean', 'weighted mean'].
            Defaults to 'weighted_mean'.
        alpha (float, optional): The significance level for the KS test. Defaults to 0.27%.
            See :py:func:`dvas.tools.gdps.stats.gdp_incompatibilities` for details.
        cws_alt_ref ('str', optional): name of the variable to use in order to generate the CWS
            alt_ref array. Holes will be interpolated linearly. Defaults to 'gph'.
        explore_covmats (bool, optional): if True, will generate plots of the covariance matrices.
            Defaults to True.

    """

    # Format the tags
    tags = dru.format_tags(start_with_tags)

    # Deal with the m_vals if warranted
    if m_vals is None:
        m_vals = [1]
    if not isinstance(m_vals, list):
        raise DvasRecipesError(f'Ouch ! m_vals should be a list of int, not: {m_vals}')

    # Get the event id and rig id
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    gdp_filt = tools.get_query_filter(tags_in=tags+[eid, rid, TAG_GDP_NAME],
                                      tags_out=dru.rsid_tags(pop=tags))
    cws_filt = tools.get_query_filter(tags_in=[eid, rid, TAG_CWS_NAME, dynamic.CURRENT_STEP_ID],
                                      tags_out=dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID))

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
        cws_tdt_exists = True
        logger.info(' ... found one !')

    except DBIOError:
        logger.info('No pre-existing CWS time profile found. Computing it now.')
        cws_tdt_exists = False

        # For the time array, let us look at the 'tdt' values of the different CWS, and choose the
        # one that starts the latest. I.e. we choose the CWS flight "start time" as the latest
        # start time of the GDPs. This is essentially a "all-or-nothing" startegy, in that we
        # assume the CWS is flying if *all* the GDPs are flying.

        # Extract the tdts
        tdts = gdp_prfs.get_prms(PRF_REF_TDT_NAME)

        # Extract the row ids that correspond to 'tdt'==0
        id_start = [int(tdts.index[tdts[col] == pd.Timedelta(0, 's')].values)
                    for col in tdts.columns]

        if len(id_start) == 0:
            logger.warning('No tdt=0 found for any GDP ?! Setting the CWS tdt=0 at step 0.')
            id_start = 0
        else:
            id_start = np.min(id_start)
            logger.info('Setting CWS tdt=0 at step: %i', id_start)

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
                                          fn_suffix=dru.fn_suffix(eid=eid, rid=rid, tags=tags,
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
    cws, covmats = dtgg.combine(gdp_prfs, binning=1, method=method,
                                mask_flgs=FLG_INCOMPATIBLE_NAME,
                                chunk_size=dynamic.CHUNK_SIZE, n_cpus=dynamic.N_CPUS)

    # Let's tag this CWS in the same way as the GDPs, so I can find them easily together
    cws.add_info_tags(tags)

    # Take a closer look at the covariance matrices, if required
    if explore_covmats:
        plots.covmat_stats(covmats)

    # Let us now hack into this cws the correct tdt, if I have it. And then save it.
    if not cws_tdt_exists:
        # tdt is an index - and replacing this is a pain. Instead, we pull it out as a column,
        # reset it, and put it back in. before saving it.
        cws[0].data.reset_index(level=PRF_REF_TDT_NAME, drop=True, inplace=True)
        new_tdt = pd.to_timedelta(np.arange(0, len(cws[0]), 1)-id_start, 's')
        cws[0].data.loc[:, PRF_REF_TDT_NAME] = pd.Series(new_tdt).values
        cws[0].data.set_index(PRF_REF_TDT_NAME, inplace=True, drop=True, append=True)

        # Having done all that, I can now save 'tdt' (and 'tdt' only, for now !) into the DB
        cws.save_to_db(add_tags=[TAG_CWS_NAME, dynamic.CURRENT_STEP_ID],
                       rm_tags=[TAG_GDP_NAME] + dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID),
                       prms=[PRF_REF_TDT_NAME])

    # Save variables of CWS to the database
    # Here, I only save the information associated to the variable, i.e. the value and its errors.
    # I do not save the alt column, which is a variable itself and should be derived as such using a
    # weighted mean. I also do not save the tdt column, which is assembled in a distinct manner.
    cws.save_to_db(add_tags=[TAG_CWS_NAME, dynamic.CURRENT_STEP_ID],
                   rm_tags=[TAG_GDP_NAME] + dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID),
                   prms=[PRF_REF_VAL_NAME, PRF_REF_UCR_NAME, PRF_REF_UCS_NAME,
                         PRF_REF_UCT_NAME, PRF_REF_UCU_NAME])

    # Deal with 'ref_alt' if warranted
    if dynamic.CURRENT_VAR == cws_alt_ref:
        # The general idea here is to take the cws gph values, and use these as the "ref_alt" of
        # the cws. Since this is used for plotting purposes, we shall fill any hole using linear
        # interpolation (holes are likely, e.g. if GDPs are incompatible, etc...!).
        # Note however that we shall only "interpolate" over holes, but will not "extrapolate"
        # beyond the outermost valid points.
        # WARNING: I am here replacing data values the hard way. This implies, in particular,
        # that the units will not be changed, nor is the reference name from the DB.
        # This will easily blow up if users don't feed a gph ...

        ref_alts = cws.get_prms(PRF_REF_VAL_NAME)
        # Quick sanity check to make sure that there is only 1 column here
        assert len(ref_alts.columns) == 1

        # Perform a linear interpolation of the holes only, using the index as anchors
        ref_alts[0].interpolate(method='index', limit_area='inside', axis=0, inplace=True)

        # Now assign these new values to the 'alt' index. Since dealing with an index is a huge pain
        # I'll extract it first
        cws[0].data.reset_index(level=PRF_REF_ALT_NAME, drop=True, inplace=True)
        cws[0].data.loc[:, PRF_REF_ALT_NAME] = ref_alts[0].values
        cws[0].data.set_index(PRF_REF_ALT_NAME, inplace=True, drop=True, append=True)

        # Excellent, I can now save this (and only this) to the DB
        cws.save_to_db(add_tags=[TAG_CWS_NAME, dynamic.CURRENT_STEP_ID],
                       rm_tags=[TAG_GDP_NAME] + dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID),
                       prms=[PRF_REF_ALT_NAME])
