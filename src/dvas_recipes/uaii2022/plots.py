"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level plotting for the UAII2022 recipe
"""

# Import general Python packages
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import dvas modules and classes
from dvas.logger import log_func_call
from dvas.data.data import MultiGDPProfile, MultiCWSProfile, MultiDeltaProfile
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME
from dvas.hardcoded import TAG_DTA_NAME, TAG_GDP_NAME, TAG_CWS_NAME
from dvas.data.data import MultiRSProfile
from dvas.plots import utils as dpu
from dvas.plots import gdps as dpg
from dvas.plots import dtas as dpd
from dvas.dvas import Database as DB

# Import from dvas_recipes
from ..errors import DvasRecipesError
from .. import dynamic
from ..recipe import for_each_flight, for_each_var
from .. import utils as dru
from . import tools

# Setup local logger
logger = logging.getLogger(__name__)


@for_each_flight
@log_func_call(logger, time_it=True)
def flight_overview(start_with_tags, label='mid', show=None):
    """ Create an "overview" plot of all the recipe variables for a given flight.

    Args:
        start_with_tags (str|list of str): tag names for the search query into the
            database. Defaults to 'sync'.
        label (str, optional): label of the plot legend. Defaults to 'mid'.
        show (bool, optional): if set, overrides the default dvas rule about whether to show the
            plot, or not. Defaults to None.

    """

    # Format the tags
    tags = dru.format_tags(start_with_tags)

    # Extract the flight info
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    filt = tools.get_query_filter(tags_in=tags+[eid, rid], tags_out=dru.rsid_tags(pop=tags))

    # The plot will have different number of rows depending on the number of variables.
    # Let's define some hardcoded heights, such that the look is always consistent
    top_gap = 0.4  # inch
    bottom_gap = 0.7  # inch
    plot_height = 1.3  # inch
    plot_gap = 0.05  # inch
    fig_height = (top_gap + bottom_gap + plot_height * len(dynamic.ALL_VARS)) /\
        (1 - plot_gap * (len(dynamic.ALL_VARS)-1))

    fig = plt.figure(figsize=(dpu.WIDTH_TWOCOL, fig_height))

    # Use gridspec for a fine control of the figure area.
    fig_gs = gridspec.GridSpec(len(dynamic.ALL_VARS), 1,
                               height_ratios=[1]*len(dynamic.ALL_VARS), width_ratios=[1],
                               left=0.085, right=0.87,
                               bottom=bottom_gap/fig_height, top=1-top_gap/fig_height,
                               wspace=0.05, hspace=plot_gap)

    for (var_ind, var_name) in enumerate(dynamic.ALL_VARS):

        # Make x a shared axis if warranted
        if var_ind > 0:
            this_ax = fig.add_subplot(fig_gs[var_ind, 0], sharex=fig.axes[0])
        else:
            this_ax = fig.add_subplot(fig_gs[var_ind, 0])

        # Reset the color cycle for each plot, so that a given RS model always has the same color
        # Adapted from the reply of pelson and gg349 on StackOverflow:
        # https://stackoverflow.com/questions/24193174
        plt.gca().set_prop_cycle(None)

        # Extract the data from the db
        rs_prfs = MultiRSProfile()
        rs_prfs.load_from_db(filt, var_name, dynamic.INDEXES[PRF_REF_TDT_NAME],
                             alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME])
        rs_prfs.sort()  # Sorting is important to make sure I have the same colors for each plot

        # Setup so dummy limits, to keep track of the actual ones as I go.
        xmin = np.infty
        xmax = -np.infty

        # Start looping over all the profiles, and plot them one-by-one.
        for (prf_ind, prf) in enumerate(rs_prfs):

            x = getattr(prf, PRF_REF_VAL_NAME).index.get_level_values(PRF_REF_TDT_NAME)
            # Transform the timedelta64[ns] in seconds
            x = x.total_seconds()
            y = getattr(prf, PRF_REF_VAL_NAME).values

            this_ax.plot(x, y, '-', lw=0.7, label='|'.join(rs_prfs.get_info(label)[prf_ind]))

            xmin = np.min([xmin, np.min(x)])
            xmax = np.max([xmax, np.max(x)])

        # Set tight x limits
        this_ax.set_xlim((xmin, xmax))

        # Once only: show the legend and the edt/eid/rid info
        if var_ind == 0:
            dpu.add_edt_eid_rid(this_ax, rs_prfs)
            dpu.fancy_legend(this_ax, label=label)
            this_ax.text(1, 1.03, 'tags: ' + ' / '.join(tags), fontsize='small',
                         verticalalignment='bottom', horizontalalignment='right',
                         transform=this_ax.transAxes)

        # Hide the tick labels except for the bottom-most plot
        if var_ind < len(dynamic.ALL_VARS)-1:
            plt.setp(this_ax.get_xticklabels(), visible=False)

        # Set the ylabel:
        ylbl = rs_prfs.var_info[PRF_REF_VAL_NAME]['prm_name']
        ylbl += ' [{}]'.format(rs_prfs.var_info[PRF_REF_VAL_NAME]['prm_unit'])
        this_ax.set_ylabel(dpu.fix_txt(ylbl), labelpad=10)

    # Set the label for the last plot only
    this_ax.set_xlabel('Time [s]')

    # Add the source
    dpu.add_source(fig)

    # Save it all
    dpu.fancy_savefig(fig, 'flight_overview', fn_prefix=dynamic.CURRENT_STEP_ID,
                      fn_suffix=dru.fn_suffix(eid=eid, rid=rid, tags=tags),
                      fmts=dpu.PLOT_FMTS, show=show)


@for_each_var
@for_each_flight
@log_func_call(logger, time_it=True)
def inspect_cws(gdp_start_with_tags, cws_start_with_tags):
    """ Create a series of CWS-related plot for inspection purposes.

    Args:
        gdp_start_with_tags (str|list of str): gdp tag names for the search query into the database.
        cws_start_with_tags (str|list of str): cws tag names for the search query into the database.
    """

    # Format the tags
    gdp_tags = dru.format_tags(gdp_start_with_tags)
    cws_tags = dru.format_tags(cws_start_with_tags)

    # This recipe step should be straight forward. I first need to extract the GDPs, then the CWS,
    # and simply call the dedicated plotting routine.

    # Get the event id and rig id
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    gdp_filt = tools.get_query_filter(tags_in=gdp_tags+[eid, rid, TAG_GDP_NAME],
                                      tags_out=dru.rsid_tags(pop=gdp_tags))
    cws_filt = tools.get_query_filter(tags_in=cws_tags+[eid, rid, TAG_CWS_NAME],
                                      tags_out=dru.rsid_tags(pop=cws_tags))

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
    # Idem for the CWS
    cws_prfs = MultiCWSProfile()
    cws_prfs.load_from_db(cws_filt, dynamic.CURRENT_VAR,
                          tdt_abbr=dynamic.INDEXES[PRF_REF_TDT_NAME],
                          alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME],
                          ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                          ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                          uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                          ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                          inplace=True)

    # We can now create a GDP vs CWS plot ...
    dpg.gdps_vs_cws(gdp_prfs, cws_prfs, show=None,
                    fn_prefix=dynamic.CURRENT_STEP_ID,
                    fn_suffix=dru.fn_suffix(eid=eid, rid=rid, tags=gdp_tags,
                                            var=dynamic.CURRENT_VAR))
    # And a uc_budget plot
    dpg.uc_budget(gdp_prfs, cws_prfs, show=None,
                  fn_prefix=dynamic.CURRENT_STEP_ID,
                  fn_suffix=dru.fn_suffix(eid=eid, rid=rid, tags=gdp_tags,
                                          var=dynamic.CURRENT_VAR))


@for_each_var
@log_func_call(logger, time_it=True)
def dtas_per_mid(start_with_tags, mids=None, skip_gdps=False, skip_nongdps=False):
    """ Create a plot of delta profiles grouped by mid.

    Args:
        start_with_tags: which tags to look for in the DB.
        mids (list, optional): list of 'mid' to process. Defaults to None = all
        skip_gdps (bool, optional): if True, any mid with 'GDP' in it will be skipped.
            Defaults to False.
        skip_nongdps (bool, optional): if True, any mid without 'GDP' will be skipped.
    """

    # Format the tags
    tags = dru.format_tags(start_with_tags)

    # Very well, let us first extract the 'mid', if they have not been provided
    db_view = DB.extract_global_view()
    if mids is None:
        mids = db_view.mid.unique().tolist()

    # Basic sanity check of mid
    if not isinstance(mids, list):
        raise DvasRecipesError('Ouch ! I need a list of mids, not: {}'.format(mids))

    # Very well, let's now loop through these, and generate the plot
    for mid in mids:

        # Second sanity check - make sure the mid is in the DB
        if mid not in db_view.mid.unique().tolist():
            raise DvasRecipesError('Ouch ! mid unknown: {}'.format(mid))

        # If warranted, skip any GDP profile
        if skip_gdps and 'GDP' in mid:
            logger.info('Skipping mid: %s', mid)
            continue

        if skip_nongdps and 'GDP' not in mid:
            logger.info('Skipping mid: %s', mid)
            continue

        # Prepare the search query
        dta_filt = tools.get_query_filter(tags_in=tags+[TAG_DTA_NAME],
                                          tags_out=dru.rsid_tags(pop=tags), mids=[mid])

        # Query these DeltaProfiles
        dta_prfs = MultiDeltaProfile()
        dta_prfs.load_from_db(dta_filt, dynamic.CURRENT_VAR,
                              alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME],
                              ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                              ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                              uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                              ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                              inplace=True)

        # Plot these deltas
        fn_suf = dru.fn_suffix(eid=None, rid=None, tags=tags, mids=[mid], var=dynamic.CURRENT_VAR)
        dpd.dtas(dta_prfs, k_lvl=1, label='mid', show=False,
                 fn_prefix=dynamic.CURRENT_STEP_ID, fn_suffix=fn_suf)
