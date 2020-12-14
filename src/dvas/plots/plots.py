"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Primary plotting functions of dvas.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..logger import log_func_call
from ..logger import plots_logger as logger
from . import utils as pu


@log_func_call(logger)
def multiprf(prfs, index_name='alt', uc=None, **kwargs):
    """ Plots the content of a MultiProfile instance.

    Args:
        prfs (MultiProfile | MultiRSProfile | MultiGDPprofile): MultiProfile instance to plot
        keys (list of str | int): list of prfs dictionnary keys to extract
        index_name (str, optional): reference variables for the plots, either 'tdt' or 'alt'.
            Defaults to 'alt'.
        uc (str, optional): which uncertainty to plot, if any. Can be one of  ['r', 's', 't', 'u',
            'tot']. Defaults to None.
        **kwargs: these get fed to the dvas.plots.utils.fancy_savefig() routine.

    """

    # Create the figure, with a suitable width.
    plt.close(10)
    fig = plt.figure(10, figsize=(pu.WIDTH_ONECOL, 5.0))

    # Use gridspec for a fine control of the figure area.
    fig_gs = gridspec.GridSpec(1, 1, height_ratios=[1], width_ratios=[1],
                               left=0.15, right=0.95, bottom=0.13, top=0.95,
                               wspace=0.05, hspace=0.05)

    # Instantiate the axes
    ax1 = fig.add_subplot(fig_gs[0, 0])
    xmin, xmax = -np.infty, np.infty

    # Do I need to extract uncertainties ?
    if uc is None:
        prms = ['val']
    elif uc == 'tot':
        prms = ['val', 'uc_tot']
    else:
        prms = ['val', 'uc%s' % (uc)]

    for prf in prfs.get_prms(prms):

        # TODO: implement the option to scale the axis with different units. E.g. 'sec' for
        # time deltas, etc ...

        # Let's extract the abscissa
        x = prf.index.get_level_values(index_name)

        # For time deltas, I need to get a float out for the limits.
        if index_name == 'tdt':
            xmin = np.nanmax([xmin, x.min(skipna=True).value])
            xmax = np.nanmin([xmax, x.max(skipna=True).value])
        else:
            xmin = np.nanmax([xmin, x.min(skipna=True)])
            xmax = np.nanmin([xmax, x.max(skipna=True)])

        # Plot the uncertainties
        if len(prms) > 1:
            ax1.fill_between(x, prf[prms[0]]-prf[prms[1]], prf[prms[0]]+prf[prms[1]],
                             alpha=0.3)
        # Plot the values
        ax1.plot(x, prf['val'].values, linestyle='-', drawstyle='steps-mid', lw=1)

    # Deal with the axes
    ax1.set_xlabel(index_name)
    ax1.set_ylabel(prfs.db_variables['val'])

    # Here, let's make sure I only ever feed floats
    ax1.set_xlim(xmin, xmax)

    # Save the plot.
    pu.fancy_savefig(fig, 'multiprf', **kwargs)
