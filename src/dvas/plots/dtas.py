"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Plotting functions related to the delta submodule.

"""

# Import from Python packages
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import from this package
from ..logger import log_func_call
from ..hardcoded import PRF_VAL, PRF_ALT, PRF_UCS, PRF_UCT, PRF_UCU, TOD_VALS
from . import utils as pu

# Setup the local logger
logger = logging.getLogger(__name__)


@log_func_call(logger)
def dtas(dta_prfs, k_lvl=1, label='mid', **kwargs):
    """ Makes a plot comparing different Delta profiles with their associated combined working
    measurement standard.

    All profiles must imperatively be fully synchronized. It is also assumed that they have all
    been build using the **same** working standard, i.e. they all have the same errors !

    Args:
        dta_prfs (dvas.data.data.MultiDeltaProfile): the Delta profiles
        k_lvl (int|float, optional): k-level for the uncertainty. Defaults to 1.
        label (str, optional): label of the plot legend. Defaults to 'mid'.
        **kwargs: these get fed to the dvas.plots.utils.fancy_savefig() routine.

    Returns:
        matplotlib.pyplot.figure: the figure instance
    """

    # Start the plotting
    fig = plt.figure(figsize=(pu.WIDTH_TWOCOL, 5.5))

    # Create a gridspec structure
    gs_info = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[1],
                                left=0.09, right=0.87, bottom=0.12, top=0.93,
                                wspace=0.5, hspace=0.1)

    # Create the axes - one for the profiles, and one for uctot, ucs, uct, ucu
    ax0 = fig.add_subplot(gs_info[0, 0])
    ax1 = fig.add_subplot(gs_info[1, 0], sharex=ax0)

    # Extract the DataFrames from the MultiGDPProfile instances
    deltas = dta_prfs.get_prms([PRF_ALT, PRF_VAL, PRF_UCS, PRF_UCT, PRF_UCU, 'uc_tot'])

    # What flights are present in the data ?
    flights = set(item + ' ' + dta_prfs.get_info('rid')[ind]
                  for (ind, item) in enumerate(dta_prfs.get_info('eid')))

    # Assess whether a single mid was provided, or not. If so, tweak some of the colors, alphas, ...
    mid = set(tuple(item) for item in dta_prfs.get_info('mid'))
    if len(mid) == 1:

        def sig_alpha(k):
            return 0.1+(3-k)*0.1

        lc = 'k'
        sig_col = 'mediumpurple'
    else:

        def sig_alpha(k):
            return 0.05+(3-k)*0.05

        lc = None
        sig_col = 'k'

    # What are the limit altitudes ?
    alt_min = deltas.loc[:, (slice(None), 'alt')].min(axis=0).min()
    alt_max = deltas.loc[:, (slice(None), 'alt')].max(axis=0).max()

    # For the bottom plots, show the k=1, 2, 3 zones
    for k in [1, 2, 3]:
        ax1.fill_between([alt_min, alt_max], [-k, -k], [k, k],
                         alpha=sig_alpha(k),
                         facecolor=sig_col, edgecolor='none')
    for ax in [ax0, ax1]:
        ax.axhline(0, lw=0.5, ls='-', c='k')

    # Very well, let us plot all these things.
    for dta_ind in deltas.columns.levels[0]:

        dta = deltas[dta_ind]

        # First, plot the profiles themselves
        if len(mid) == 1:

            ax0.fill_between(dta.loc[:, PRF_ALT],
                             dta.loc[:, PRF_VAL].values - k_lvl * dta.loc[:, 'uc_tot'].values,
                             dta.loc[:, PRF_VAL].values + k_lvl * dta.loc[:, 'uc_tot'].values,
                             alpha=1/len(flights), step='mid', facecolor='k', edgecolor='none')

        else:
            # Plot the uncertainties of the CWS ...
            if dta_ind == 0:
                # ... or not. Turns out, the CWS uncertainty is not always the same. If one point
                # is a NaN, then it has a NaN uncertainty, which implies the simple test initially
                # implemented to check that all the uc are the same does not work.
                # Rather than doing anything too complicated, let's just avoid showing the CWS
                # in the plot.
                pass
                #ax0.fill_between(dta.loc[:, PRF_ALT],
                #                 - k_lvl * dta.loc[:, 'uc_tot'].values,
                #                 + k_lvl * dta.loc[:, 'uc_tot'].values,
                #                 alpha=0.2, step='mid', facecolor='k', edgecolor='none',
                #                 label='CWS')
            #else:
            #    if not dta.loc[:, 'uc_tot'].equals(deltas[0].loc[:, 'uc_tot']):
            #        logger.error(
            #            'Inconsistent delta uncertainties will not be reflected in the plot.')

            # ... and the delta curves themsleves
            ax0.plot(dta.loc[:, PRF_ALT].values, dta.loc[:, PRF_VAL].values,
                     lw=0.4, ls='-', drawstyle='steps-mid', c=lc, alpha=1,
                     label='|'.join(dta_prfs.get_info(label)[dta_ind]))

        # And then, the deltas normalized by the uncertainties
        ax1.plot(dta.loc[:, PRF_ALT], dta.loc[:, PRF_VAL] / dta.loc[:, 'uc_tot'].values,
                 lw=0.5, ls='-', drawstyle='steps-mid', c=lc, alpha=1/len(flights))

    # Set the axis labels
    ylbl0 = r'$\delta_{e,i}$'
    ylbl0 += f' [{dta_prfs.var_info[PRF_VAL]["prm_unit"]}]'
    ylbl1 = r'$\delta_{e,i}/\sigma(\delta_{e,i})$'
    #altlbl = r'gph$_{\rm CWS}$'
    altlbl = f'{dta_prfs.var_info[PRF_ALT]["prm_plot"]} [{dta_prfs.var_info[PRF_ALT]["prm_unit"]}]'

    # Plot ylabel as text, to have them left aligned accross subplots
    ax0.text(-0.1, 0.5, pu.fix_txt(ylbl0), ha='left', va='center',
             transform=ax0.transAxes, rotation=90)
    ax1.text(-0.1, 0.5, pu.fix_txt(ylbl1), ha='left', va='center',
             transform=ax1.transAxes, rotation=90)
    ax1.set_xlabel(pu.fix_txt(altlbl))

    # Hide certain ticks, and set the limits
    ax1.set_ylim((-6, +6))
    plt.setp(ax0.get_xticklabels(), visible=False)

    if len(mid) > 1:
        # Add the legend
        pu.fancy_legend(ax0, label)
        # Add the edt/eid/rid info
        pu.add_edt_eid_rid(ax0, dta_prfs)
    elif len(mid) == 1:
        # Add the number of flights and time of day to the plot. Will be useful with large numbers
        # TODO: do not assume that I need to drop the first 4 char from tod ... know it !
        tod = ' + '.join([item[4:] for item in TOD_VALS if any(dta_prfs.has_tag(item))])
        ax0.text(0, 1.03, rf'\# flights: {len(flights)} [{tod}]', fontsize='small',
                 verticalalignment='bottom', horizontalalignment='left',
                 transform=ax0.transAxes)

    # Add the source for the plot
    pu.add_source(fig)

    # Add the variable name and k level
    if len(mid) == 1:
        mid_msg = '-'.join(list(mid)[0])
    else:
        mid_msg = None
    pu.add_var_and_k(ax0, mid=mid_msg, var_name=dta_prfs.var_info[PRF_VAL]['prm_plot'], k=k_lvl)

    # Save it
    pu.fancy_savefig(fig, fn_core='dtas', **kwargs)
