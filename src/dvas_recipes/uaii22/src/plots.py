"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level plotting for the UAII22 recipe
"""

# Import general Python packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import dvas modules and classes
from dvas.logger import recipes_logger as logger
from dvas.logger import log_func_call
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME
from dvas.data.data import MultiRSProfile
from dvas.plots import utils as dpu

# Import from dvas_recipes
from ...errors import DvasRecipesError
from ... import dynamic
from ...recipe import for_each_flight

@for_each_flight
@log_func_call(logger, time_it=True)
def flight_overview(tags='sync', label='mid'):
    """ Create an "overview" plot of all the recipe variables for a given flight.

    Args:
        tags (str|list of str, optional): tag names for the search query into the database.
            Defaults to 'sync'.

    """

    # Deal with the search tags
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        raise DvasRecipesError('Ouch ! tags should be of type str|list. not: {}'.format(type(tags)))

    # Extract the flight info
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    filt = "and_(tags('e:{}'), tags('r:{}'), {})".format(eid, rid,
                                                         "tags('" + "'), tags('".join(tags) + "')")

    # The plot will have different number of rows depending on the number of variables.
    # Let's define some hardcoded heights, such that the look is always consistent
    top_gap = 0.4 # inch
    bottom_gap = 0.7 # inch
    plot_height = 1.3 # inch
    plot_gap = 0.05 # inch
    fig_height = (top_gap + bottom_gap + plot_height * len(dynamic.ALL_VARS))/\
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
        rs_prfs.sort() # Sorting is important to make sure I have the same colors for each plot

        #Setup so dummy limits, to keep track of the actual ones as I go.
        xmin = np.infty
        xmax = -np.infty

        # Start looping over all the profiles, and plot them one-by-one.
        for (prf_ind, prf) in enumerate(rs_prfs):

            x = getattr(prf, PRF_REF_VAL_NAME).index.get_level_values(PRF_REF_TDT_NAME)
            # Transform the timedelta64[ns] in seconds
            x = x.total_seconds()
            y = getattr(prf, PRF_REF_VAL_NAME).values

            this_ax.plot(x, y, '-', lw=0.7, label=rs_prfs.get_info(label)[prf_ind])

            xmin = np.min([xmin, np.min(x)])
            xmax = np.max([xmax, np.max(x)])

        # Set tight x limits
        this_ax.set_xlim((xmin, xmax))

        # Once only: show the legend and the edt/eid/rid info
        if var_ind == 0:
            dpu.add_edt_eid_rid(this_ax, rs_prfs)
            dpu.fancy_legend(this_ax, label=label)
            # TODO: create a specific function for random text like below ?
            this_ax.text(1, 1.03, 'tags: ' + ' / '.join(tags), fontsize='small',
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=this_ax.transAxes)

        # Hide the tick labels except for the bottom-most plot
        if var_ind < len(dynamic.ALL_VARS)-1:
            plt.setp(this_ax.get_xticklabels(), visible=False)

        # Set the ylabel:
        ylbl = rs_prfs.var_info[PRF_REF_VAL_NAME]['prm_name']
        ylbl += ' [{}]'.format(rs_prfs.var_info[PRF_REF_VAL_NAME]['prm_unit'])
        this_ax.set_ylabel(ylbl, labelpad=10)

    # Set the label for the last plot only
    this_ax.set_xlabel('Time [s]')

    # Add the source
    dpu.add_source(fig)

    import pdb
    pdb.set_trace()

    # Save it all
    dpu.fancy_savefig(fig, 'flight_overview', fn_prefix=dynamic.CURRENT_STEP_ID,
                      fn_suffix='e{}_r{}_{}'.format(eid, rid, '-'.join(tags)),
                      fmts=dpu.PLOT_FMTS, show=dpu.PLOT_SHOW)
