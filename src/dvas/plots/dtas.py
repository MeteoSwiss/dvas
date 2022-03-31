"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Plotting functions related to the delta submodule.

"""

# Import from Python packages
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import from this package
from ..logger import log_func_call
from ..errors import DvasError
from ..hardcoded import PRF_REF_INDEX_NAME, PRF_REF_VAL_NAME, PRF_REF_ALT_NAME, PRF_REF_TDT_NAME
from ..hardcoded import PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME
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

    # Create the axes - one for the profiles, and one for uctot, ucr, ucs, uct, ucu
    ax0 = fig.add_subplot(gs_info[0, 0])
    ax1 = fig.add_subplot(gs_info[1, 0], sharex=ax0)

    # Extract the DataFrames from the MultiGDPProfile instances
    dtas = dta_prfs.get_prms([PRF_REF_ALT_NAME, PRF_REF_VAL_NAME, PRF_REF_UCR_NAME,
                              PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME,
                              'uc_tot'])

    # What are the limit altitudes ?
    alt_min = dtas.loc[:, (slice(None), 'alt')].min(axis=0).min()
    alt_max = dtas.loc[:, (slice(None), 'alt')].max(axis=0).max()

    # For the bottom plots, show the k=1, 2, 3 zones
    for k in [1, 2, 3]:
        ax1.fill_between([alt_min, alt_max], [-k, -k], [k, k],
                         alpha=0.05+(3-k)*0.05,
                         facecolor='k', edgecolor='none')
    for ax in [ax0, ax1]:
        ax.axhline(0, lw=0.5, ls='-', c='k')

    # Very well, let us plot all these things.
    for dta_ind in dtas.columns.levels[0]:

        dta = dtas[dta_ind]

        # First, plot the profiles themselves
        ax0.plot(dta.loc[:, PRF_REF_ALT_NAME].values, dta.loc[:, PRF_REF_VAL_NAME].values,
                 lw=0.5, ls='-', drawstyle='steps-mid',
                 label='|'.join(dta_prfs.get_info(label)[dta_ind]))

        # Next plot the uncertainties
        ax0.fill_between(dta.loc[:, PRF_REF_ALT_NAME],
                         dta.loc[:, PRF_REF_VAL_NAME] - k_lvl * dta.loc[:, 'uc_tot'].values,
                         dta.loc[:, PRF_REF_VAL_NAME] + k_lvl * dta.loc[:, 'uc_tot'].values,
                         alpha=0.3, step='mid')

        # And then, the deltas normalized by the uncertainties
        ax1.plot(dta.loc[:, PRF_REF_ALT_NAME],
                 dta.loc[:, PRF_REF_VAL_NAME] / dta.loc[:, 'uc_tot'].values,
                 lw=0.5, ls='-', drawstyle='steps-mid')

    # Set the axis labels
    ylbl0 = r'$\delta_{e,i}$'
    ylbl0 += ' [{}]'.format(dta_prfs.var_info[PRF_REF_VAL_NAME]['prm_unit'])
    ylbl1 = r'$\delta_{e,i}/\sigma_{\Omega_{e,i}}$'
    altlbl = dta_prfs.var_info[PRF_REF_ALT_NAME]['prm_name']
    altlbl += ' [{}]'.format(dta_prfs.var_info[PRF_REF_ALT_NAME]['prm_unit'])

    ax0.set_ylabel(pu.fix_txt(ylbl0), labelpad=10)
    ax1.set_ylabel(pu.fix_txt(ylbl1), labelpad=10)
    ax1.set_xlabel(pu.fix_txt(altlbl))

    # Hide certain ticks, and set the limits
    ax0.set_xlim((alt_min, alt_max))
    ax1.set_ylim((-5, +5))
    plt.setp(ax0.get_xticklabels(), visible=False)

    # Add the legend
    pu.fancy_legend(ax0, label)

    # Add the edt/eid/rid info
    pu.add_edt_eid_rid(ax0, dta_prfs)

    # Add the k-level and variable name
    ax0.text(1, 1.03, r'{}, $k={}$'.format(dta_prfs.var_info[PRF_REF_VAL_NAME]['prm_name'], k_lvl),
             fontsize='small',
             verticalalignment='bottom', horizontalalignment='right',
             transform=ax0.transAxes)

    # Add the source for the plot
    pu.add_source(fig)

    # Save it
    pu.fancy_savefig(fig, fn_core='dtas', **kwargs)
