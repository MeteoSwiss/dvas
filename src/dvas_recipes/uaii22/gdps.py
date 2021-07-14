"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level GDP recipes for the UAII22 campaign
"""

# Import from dvas
from dvas.data.data import MultiRSProfile, MultiGDPProfile
from dvas.tools.gdps import stats as dtgs
from dvas.tools.gdps import gdps as dtgg
import dvas.plots.gdps as dpg
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, FLG_INCOMPATIBLE_NAME
from dvas.errors import DBIOError

# Import from dvas_recipes
from ..errors import DvasRecipesError
from ..recipe import for_each_flight, for_each_var
from .. import dynamic
from ..utils import fn_suffix

@for_each_var
@for_each_flight
def build_cws(tags='sync', m_vals=[1, '2*'], strategy='all-or-none'):
    """ Highest-level recipe function responsible for assembling the combined working standard for
    a specific RS flight.

    This function directly builds the profiles and upload them to the db with the 'cws' tag.

    Args:
        tags (str|list of str, optional): tag name(s) for the search query into the database.
            Defaults to 'sync'.
        m_vals (int|list, optional): list of m-values used for identifiying incompatible and valid
            regions between GDPs. Any value listed as 'x*' will be ignored when computing the cws.
        strategy (str, optional): name of GDP combination strategy (for deciding which levels/
            measurements are valid or not). Defaults to 'all-or-none'. These ared defined in
            `dvas.tools.gdps.stats.get_validities()`.

    """

    # Deal with the search tags
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        raise DvasRecipesError('Ouch ! tags should be of type str|list. not: {}'.format(type(tags)))

    # Deal with the m_vals if warranted
    if isinstance(m_vals, str):
        # Here, let's just deconstruct the string into components, but keep everything as str
        m_vals = m_vals.split(' ')

    # Get the event id and rig id
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    gdp_filt = "and_(tags('gdp'), tags('e:{}'), tags('r:{}'), {})".format(eid, rid,
        "tags('" + "'), tags('".join(tags) + "')")
    cws_filt = "and_(tags('cws'), tags('e:{}'), tags('r:{}'), {})".format(eid, rid,
        "tags('" + "'), tags('".join(tags) + "')")

    # First things first, let's check if we have already computed the 'tdt' profile of the CWS
    # during a previous pass. If not, create it now. We treat the case of the tdt sepearately,
    # since we compute it as a normal (unweighted) mean.
    try:
        _ = MultiRSProfile().load_from_db(cws_filt, 'time', 'time')

    except DBIOError:
        # Very well, let us compute the time array of the combined working standard. Since there is
        # no uncertainty associated to it, we shall simply take it as the arithmetic mean of the
        # individual GDP times.
        # First, load the time data

        gdp_prfs = MultiGDPProfile()
        gdp_prfs.load_from_db(gdp_filt, dynamic.INDEXES[PRF_REF_TDT_NAME],
                              tdt_abbr=dynamic.INDEXES[PRF_REF_TDT_NAME],
                              alt_abbr=None, ucr_abbr=None, ucs_abbr=None,
                              uct_abbr=None, ucu_abbr=None,
                              inplace=True)

        import pdb
        pdb.set_trace()

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



    # Before combining the GDPs with each other, let us assess their consistency.
    # The idea here is to flag any inconsistent measurement, so that they can be ignored during
    # the combination process.
    incompat = dtgs.gdp_incompatibilities(gdp_prfs, alpha=0.0027,
                                          m_vals=[int(item.replace('*', '')) for item in m_vals],
                                          do_plot=True,
                                          n_cpus=dynamic.N_CPUS, fn_prefix=dynamic.CURRENT_STEP_ID,
                                          fn_suffix=fn_suffix(eid=eid, rid=rid, tags=tags,
                                                              var=dynamic.CURRENT_VAR))

    # Next, we derive "validities" given a specific strategy to assess the different GDP pair
    # incompatibilities ...
    # Note how the m_vals used for the combination can differ from the ones used to check the
    # incompatibilities. This is intended to let people experiment a bit without affecting the final
    # CWS.
    valids = dtgs.gdp_validities(incompat,
                                 m_vals=[int(item) for item in m_vals if '*' not in item],
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
    dpg.gdps_vs_cws(gdp_prfs, cws, index_name='_idx', show=True, fn_prefix=dynamic.CURRENT_STEP_ID,
                    fn_suffix=fn_suffix(eid=eid, rid=rid, tags=tags, var=dynamic.CURRENT_VAR))

    # Save the CWS to the database
    # Here, I only save the information associated to the variable, i.e. the value and its errors.
    # I do not save the alt column, which is a variable itself and should be derived as such using a
    # weighted mean. I also do not save the tdt column, which should be assembled from a simple mean
    cws.save_to_db(add_tags=['cws'], rm_tags=['gdp'], prms=['val', 'ucr', 'ucs', 'uct', 'ucu'])
