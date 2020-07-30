"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Primary plotting functions of dvas.

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..dvas_environ import path_var as env_path_var
from . import plot_utils as pu


def basic_plot(items, fig_num=None, save_fn=None):
    """ Create a basic plot.

    Args:
        items (list): list of TimeProfileManager
        fig_num (int, optional): figure number
        save_fn (str, optional): name of the plot file to save. If None, no plot is saved.

    """

    # Instantiate the figure, closing it first if it already exists.
    if fig_num is not None:
        plt.close(fig_num)

    fig = plt.figure(fig_num, figsize=(pu.WIDTH_ONECOL, 5.0))

    # USe gridspec for a fine control of the figure area.
    fig_gs = gridspec.GridSpec(1, 1, height_ratios=[1], width_ratios=[1],
                               left=0.15, right=0.95, bottom=0.13, top=0.95,
                               wspace=0.05, hspace=0.05)

    # Instantiate the axes
    ax1 = plt.subplot(fig_gs[0, 0])
    xmin, xmax = 0, np.infty

    for arg in items:
        # Extract the timedeltas in seconds
        x_data_s = arg.data.index.astype('timedelta64[s]')

        xmin = np.nanmax([xmin, np.min(x_data_s)])
        xmax = np.nanmin([xmax, np.max(x_data_s)])

        ax1.plot(x_data_s, arg.data.values, linestyle='-', drawstyle='steps-mid')

    # Deal with the axes
    ax1.set_xlabel(r'$\Delta t$ [s]')
    # TODO: Also add the y label

    ax1.set_xlim(xmin, xmax)

    # If requested, save the plot.
    if save_fn is not None:
        for this_type in pu.PLOT_TYPES:
            if this_type not in fig.canvas.get_supported_filetypes().keys():
            # TODO: log this as a warning: request style not available
                pass

            # Save the file.
            # Note: never use a tight box fix here. If the plot looks weird, the gridspec
            # params should be altered. THis is essential for the consistency of the DVAS plots.
            plt.savefig(Path(env_path_var.output_path, save_fn+this_type))

    # Show the plot, or just close it and move on
    if pu.PLOT_SHOW:
        plt.show()
    else:
        plt.close(fig_num)
