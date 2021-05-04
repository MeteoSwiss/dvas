"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Primary plotting functions of dvas.

"""
# import from python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# import from dvas
from ..errors import DvasError
from ..hardcoded import PRF_REF_VAL_NAME, PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME
from ..hardcoded import PRF_REF_UCU_NAME
from ..logger import log_func_call
from ..logger import plots_logger as logger
from . import utils as pu


@log_func_call(logger)
def multiprf(prfs, index='alt', label='mid', uc=None, k_lvl=1, **kwargs):
    """ Plots the content of a MultiProfile instance.

    Args:
        prfs (MultiProfile|MultiRSProfile|MultiGDPprofile): MultiProfile instance to plot
        index (str, optional): reference variables for the plots, either '_idx', 'tdt' or 'alt'.
            Defaults to 'alt'.
        label (str, optional): name of the label for each curve, that will be fed to
            `prfs.get_info(label)`. Defaults to 'mid'.
        uc (str, optional): which uncertainty to plot, if any. Can be one of  ['ucr', 'ucs', 'uct',
            'ucu', 'uc_tot']. Defaults to None.
        k_lvl (int|float, optional): k-level for the uncertainty, if uc is not None. Defaults to 1.
        **kwargs: these get fed to the dvas.plots.utils.fancy_savefig() routine.

    Returns:
        matplotlib.pyplot.figure: the figure instance

    """

    # Some sanity checks
    if uc not in [None, 'uc_tot', PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME,
                  PRF_REF_UCU_NAME]:
        raise DvasError('Ouch ! Unknown uc name: {}'.format(uc))

    if not isinstance(k_lvl, (float, int)):
        raise DvasError(
            'Ouch ! Bad type for k_lvl. Needed int| float - got: {}'.format(type(k_lvl)))

    # Create the figure, with a suitable width.
    fig = plt.figure(figsize=(pu.WIDTH_TWOCOL, 4.0))

    # Use gridspec for a fine control of the figure area.
    fig_gs = gridspec.GridSpec(1, 1, height_ratios=[1], width_ratios=[1],
                               left=0.08, right=0.87, bottom=0.15, top=0.9,
                               wspace=0.05, hspace=0.05)

    # Instantiate the axes
    ax1 = fig.add_subplot(fig_gs[0, 0])
    xmin, xmax = -np.infty, np.infty

    for (p_ind, prf) in enumerate(prfs):

        # TODO: implement the option to scale the axis with different units. E.g. 'sec' for
        # time deltas, etc ...

        # Let's extract the data
        x = getattr(prf, PRF_REF_VAL_NAME).index.get_level_values(index)
        y = getattr(prf, PRF_REF_VAL_NAME).values
        if uc is not None:
            dy = getattr(prf, uc).values * k_lvl

        # For time deltas, I need to get a float out for the limits.
        if index == 'tdt':
            x = x.total_seconds()

        xmin = np.nanmax([xmin, x.min(skipna=True)])
        xmax = np.nanmin([xmax, x.max(skipna=True)])

        # Plot the uncertainties if requested
        if uc is not None:
            ax1.fill_between(x, y-dy, y+dy, alpha=0.3)

        # Plot the values
        ax1.plot(x, y, linestyle='-', drawstyle='steps-mid', lw=1,
                 label=prfs.get_info(label)[p_ind])

    # Deal with the axes
    ax1.set_xlabel(index)
    ax1.set_ylabel(prfs.db_variables[PRF_REF_VAL_NAME], labelpad=10)
    ax1.set_xlim(xmin, xmax)

    # Add the legend
    pu.fancy_legend(ax1, label)

    # Add the edt/eid/rid info
    pu.add_edt_eid_rid(ax1, prfs)

    # Add the source for the plot
    pu.add_source(fig)

    # Save the plot
    pu.fancy_savefig(fig, 'multiprf', **kwargs)

    return fig
