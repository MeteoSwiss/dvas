"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level GDP recipes for the UAII22 campaign
"""

# Import from dvas
from dvas.data.data import MultiGDPProfile
from dvas.tools.gdps import stats as dtgs
from dvas.tools.gdps import gdps as dtgg
import dvas.plots.gdps as dpg
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME

# Import from dvas_recipes
from ..errors import DvasRecipesError
from ..recipe import for_each_flight, for_each_var
from .. import dynamic
from ..utils import fn_suffix

@for_each_var
@for_each_flight
def build_cws(tags='sync'):
    """ Highest-level recipe function responsible for assembling the combined working standard for
    a specific RS flight.

    This function directly builds the profiles and upload them to the db with the 'cws' tag.

    Args:
        tags (str|list of str, optional): tag name(s) for the search query into the database.
            Defaults to 'sync'.

    """

    # Deal with the search tags
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        raise DvasRecipesError('Ouch ! tags should be of type str|list. not: {}'.format(type(tags)))

    # Get the event id and rig id
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    filt = "and_(tags('gdp'), tags('e:{}'), tags('r:{}'), {})".format(eid, rid,
        "tags('" + "'), tags('".join(tags) + "')")

    # Load the GDP profiles
    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(filt, dynamic.CURRENT_VAR,
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
    out = dtgs.get_incompatibility(gdp_prfs, alpha=0.0027, bin_sizes=[1, 2, 4], do_plot=True,
                                   n_cpus=dynamic.N_CPUS, fn_prefix=dynamic.CURRENT_STEP_ID,
                                   fn_suffix=fn_suffix(eid=eid, rid=rid, tags=tags,
                                                       var=dynamic.CURRENT_VAR))

    # TODO: set flags based on the incompatibilities derived.

    # Let us now create a high-resolution CWS for these synchronized GDPs
    cws = dtgg.combine(gdp_prfs, binning=1, method='weighted mean', chunk_size=dynamic.CHUNK_SIZE,
                       n_cpus=dynamic.N_CPUS)

    # We can now inspect the result visually
    dpg.gdps_vs_cws(gdp_prfs, cws, index_name='_idx', show=True, fn_prefix=dynamic.CURRENT_STEP_ID,
                    fn_suffix=fn_suffix(eid=eid, rid=rid, tags=tags, var=dynamic.CURRENT_VAR))

    # --- TODO ---
    # Save the CWS to the database
    #cws.save_to_db()
