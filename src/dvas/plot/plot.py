"""
Module containing the primary plotting functions for dvas.

Created June 2020; F.P.A. Vogt; frederic.vogt@meteoswiss.ch
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..dvas_environ import path_var as env_path_var
from ..dvas_environ import package_path


# Define some constants
#: float: Width of a 1-column plot [inches], to fit in scientific articles when scaled by 50%
WIDTH_ONECOL = 6.92

#: float: Width of a 2-column plot [inches], to fit in scientific articles when scaled by 50%
WIDTH_TWOCOL = 14.16

#: dict: The name of the different dvas matplotlib style sheets
PLOT_STYLES = {'base': 'base.mplstyle',
               'nolatex': 'nolatex.mplstyle',
               'latex': 'latex.mplstyle'}

# Immediately enable the base look for DVAS plots, as soon as we load the module.
plt.style.use(str(Path(package_path, 'plot', 'mpl_styles', PLOT_STYLES['base'])))

#: list[str]: The default file extensions to save the plots into.
PLOT_TYPES = ['.png', '.pdf']

# A flag to display the plots or not.
PLOT_SHOW = True


def set_mplstyle(style='base'):
    """ Set the DVAS plotting style. 'base' contains all the generic commands. 'latex'
    enables the use of a system-wide LaTeX engine. 'nolatex' disables it.

    Args:
        name (str, dict; optional): A style specification. Defaults to 'base'. Valid options are:

            - *str*: One of ['base', 'nolatex', 'latex'] defined inside DVAS.
            - *dict*: Dictionary with valid key/value pairs for matplotlib.rcParams.

    Caution:
        Specifying the 'latex' style requires a working system-wide LaTeX installation.
        Specifying 'nolatex' enables the default matplotlib LaTeX.

    """

    # Let's start with some sanity checks. If the user is foolish enough to feed a dict, trust it.
    if isinstance(style, dict):
        plt.style.use(style)

    # Else, if this is not a known str, let's be unforgiving.
    if not isinstance(style, str):
        raise Exception('Ouch ! style type must be one of (str, dict), not: %s' % (type(style)))

    if style not in PLOT_STYLES.keys():
        raise Exception('Ouch! plot style "%s" unknown. Should be one of [%s].'
                        % (style, ', '.join(PLOT_STYLES.keys())))

    # Always apply the base style first.
    plt.style.use(str(Path(package_path, 'plot', 'mpl_styles', PLOT_STYLES['base'])))

    # Then apply which ever alternative style was requested, if we haven't already.
    if style != 'base':
        plt.style.use(str(Path(package_path, 'plot', 'mpl_styles', PLOT_STYLES[style])))


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

    fig = plt.figure(fig_num, figsize=(WIDTH_ONECOL, 5.0))

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
        for this_type in PLOT_TYPES:
            if this_type not in fig.canvas.get_supported_filetypes().keys():
            # TODO: log this as a warning: request style not available
                pass

            # Save the file.
            # Note: never use a tight box fix here. If the plot looks weird, the gridspec
            # params should be altered. THis is essential for the consistency of the DVAS plots.
            plt.savefig(Path(env_path_var.output_path, save_fn+this_type))

    # Show the plot, or just close it and move on
    if PLOT_SHOW:
        plt.show()
    else:
        plt.close(fig_num)
