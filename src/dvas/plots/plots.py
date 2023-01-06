"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Primary plotting functions of dvas.

"""
# import from python
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# import from dvas
from ..errors import DvasError
from ..hardcoded import PRF_VAL, PRF_UCS, PRF_UCT, PRF_UCU
from ..logger import log_func_call
from . import utils as pu

# Setup local logger
logger = logging.getLogger(__name__)


@log_func_call(logger)
def multiprf(prfs, index='alt', label='mid', uc=None, k_lvl=1, rel_to=None, expose=None, **kwargs):
    """ Plots the content of a MultiProfile instance.

    Args:
        prfs (MultiProfile|MultiRSProfile|MultiGDPprofile): MultiProfile instance to plot
        index (str, optional): reference variables for the plots, either '_idx', 'tdt' or 'alt'.
            Defaults to 'alt'.
        label (str, optional): name of the label for each curve, that will be fed to
            `prfs.get_info(label)`. Defaults to 'mid'.
        uc (str, optional): which uncertainty to plot, if any. Can be one of  ['ucs', 'uct',
            'ucu', 'uc_tot']. Defaults to None.
        k_lvl (int|float, optional): k-level for the uncertainty, if uc is not None. Defaults to 1.
        rel_to (int, optional): if set, will plot the differences with respect to prfs[rel_to].
            Defaults to None. If set, the profiles MUST have been synchronized beforehand !
        expose (int, optional): if set, only profile[expose] will be plotted in color.
            This is meant to "anonymize" the plot with respect to a specific profile. Defaults to
            None. If set, the labels of *all* the profiles will still appear !
        **kwargs: these get fed to the dvas.plots.utils.fancy_savefig() routine.

    Returns:
        matplotlib.pyplot.figure: the figure instance

    """

    # Some sanity checks
    if uc not in [None, 'uc_tot', PRF_UCS, PRF_UCT, PRF_UCU]:
        raise DvasError('Unknown uc name: {uc}')

    if not isinstance(k_lvl, (float, int)):
        raise DvasError(f'Bad type for k_lvl. Needed int| float - got: {type(k_lvl)}')

    if rel_to is not None:
        if not isinstance(rel_to, int):
            raise DvasError(f'rel_to should be of type int, not: {type(rel_to)}')
        if rel_to < 0 or rel_to >= len(prfs):
            raise DvasError(f'rel_to must be in range [0, {len(prfs)-1}]')
        if len(set([len(item) for item in prfs])) > 1:
            raise DvasError('rel_to requires all profiles to have the same length.')

    if expose is not None:
        if not isinstance(expose, int):
            raise DvasError(f'expose should be of type int, not: {type(expose)}')
        if expose < 0 or expose >= len(prfs):
            raise DvasError(f'expose must be in range [0, {len(prfs)-1}]')

    # Create the figure, with a suitable width.
    fig = plt.figure(figsize=(pu.WIDTH_TWOCOL, 4.0))

    # Use gridspec for a fine control of the figure area.
    fig_gs = gridspec.GridSpec(1, 1, height_ratios=[1], width_ratios=[1],
                               left=0.08, right=0.87, bottom=0.17, top=0.9,
                               wspace=0.05, hspace=0.05)

    # Instantiate the axes
    ax1 = fig.add_subplot(fig_gs[0, 0])
    xmin, xmax = -np.infty, np.infty

    # Do I have a reference profile ?
    if rel_to is None:
        ref_prfs = [np.zeros(len(item)) for item in prfs]
    else:
        ref_prfs = [getattr(prfs[rel_to], PRF_VAL)] * len(prfs)

    # Let's extract the color cycler, so that I can properly expose a specific profile if I have to.
    cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for (p_ind, prf) in enumerate(prfs):

        # Get the color from the cycler if warranted ...
        if expose is None:
            clr = cycler[p_ind]
        elif expose == p_ind:
            clr = pu.CLRS['ref_1']
        else:
            clr = pu.CLRS['nan_1']

        # Let's extract the data
        x = getattr(prf, PRF_VAL).index.get_level_values(index)
        y = getattr(prf, PRF_VAL).values - ref_prfs[p_ind]
        if uc is not None:
            dy = getattr(prf, uc).values * k_lvl

        # For time deltas, I need to get a float out for the limits.
        if index == 'tdt':
            x = x.total_seconds()

        xmin = np.nanmax([xmin, x.min(skipna=True)])
        xmax = np.nanmin([xmax, x.max(skipna=True)])

        # Plot the uncertainties if requested
        if uc is not None:
            ax1.fill_between(x, y-dy, y+dy, alpha=0.3, step='mid', color=clr)

        # Plot the values
        ax1.plot(x, y, linestyle='-', drawstyle='steps-mid', lw=1, color=clr,
                 label=prfs.get_info(label)[p_ind])

    # Deal with the axes labels
    xlbl = prfs.var_info[index]['prm_name']
    xlbl += f" [{prfs.var_info[index]['prm_unit']}]"
    ax1.set_xlabel(pu.fix_txt(xlbl))

    ylbl = prfs.var_info[PRF_VAL]['prm_name']
    ylbl += f" [{prfs.var_info[PRF_VAL]['prm_unit']}]"
    if rel_to is not None:
        ylbl = r'$\Delta$' + ylbl
    ax1.set_ylabel(pu.fix_txt(ylbl), labelpad=10)

    ax1.set_xlim(xmin, xmax)

    # Add the legend
    pu.fancy_legend(ax1, label)

    # Add the edt/eid/rid info
    pu.add_edt_eid_rid(ax1, prfs)

    # Add the k-level if warranted
    if uc is not None:
        ax1.text(1, 1.03, r'$k={}$'.format(k_lvl), fontsize='small',
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=ax1.transAxes)

    # Add the source for the plot
    pu.add_source(fig)

    # Save the plot
    pu.fancy_savefig(fig, 'multiprf', **kwargs)

    return fig
