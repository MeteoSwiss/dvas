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

# Import from dvas_recipes
from ...errors import DvasRecipesError
from ... import dynamic

def build_cws(evt_id, rig_id, rcp_vars, tags='sync', step_id=None):
    """ Highest-level function responsible for assembling the combined working standard for a
    specific RS flight.

    This function directly builds the profiles and upload them to the db with the 'cws' tag.

    Args:
        evt_id (str|int): event id to be synchronized, e.g. 80611
        rig_id (str|int): rig id to be synchronized, e.g. 1
        rcp_vars (dict): names of variables to process, and associated uncertainties, e.g.::

            {'temp': {'ucr': 'temp_ucr', 'ucs': 'temp_ucs', 'uct': 'temp_uct', 'ucu':}}

        tags (str|list of str, optional): tag names for the search query into the database.
            Defaults to 'sync'.
        step_id (int|str): recipe step id, to be added at thew front of the resulting plot file.
            Defaults to None.

    """

    # Deal with the search tags
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        raise DvasRecipesError('Ouch ! tags should be of type str|list. not: {}'.format(type(tags)))

    # What search query will let me access the data I need ?
    filt = "and_(tags('gdp'), tags('e:{}'), tags('r:{}'), {})".format(evt_id, rig_id,
                                                         "tags('" + "'), tags('".join(tags) + "')")


    for (var_ind, var_name) in enumerate(rcp_vars):

        gdp_prfs = MultiGDPProfile()
        gdp_prfs.load_from_db(filt, var_name, tdt_abbr='time', alt_abbr='gph',
                              ucr_abbr=rcp_vars[var_name]['ucr'],
                              ucs_abbr=rcp_vars[var_name]['ucs'],
                              uct_abbr=rcp_vars[var_name]['uct'],
                              ucu_abbr=rcp_vars[var_name]['ucu'],
                              inplace=True)

        # Before combining the GDPs with each other, let us assess their consistency.
        # The idea here is to flag any inconsistent measurement, so that they can be ignored during
        # the combination process.
        out = dtgs.get_incompatibility(gdp_prfs, alpha=0.0027, bin_sizes=[1, 2, 4, 8], do_plot=True,
                                       n_cpus=8)

        # TODO: set flags based on the incompatibilities derived.

        # Let us now create a high-resolution CWS for these synchronized GDPs
        cws = dtgg.combine(gdp_prfs, binning=1, method='weighted mean', chunk_size=200, n_cpus=8)

        # We can now inspect the result visually
        dpg.gdps_vs_cws(gdp_prfs, cws, index_name='_idx', show=True, fn_prefix=step_id,
                        fn_suffix='e{}_r{}_{}_{}'.format(evt_id, rig_id, '-'.join(tags), var_name))

        # --- TODO ---
        # Save the CWS to the database
        #cws.save_to_db()
