"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Utility functions and parameters for plotting in dvas.

"""

# Import from Python packages
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

# Import from this package
from .. import pkg_path
from ..dvas_logger import plot_logger, log_func_call

# Define some plotting constants
#: float: Width of a 1-column plot [inches], to fit in scientific articles when scaled by 50%
WIDTH_ONECOL = 6.92

#: float: Width of a 2-column plot [inches], to fit in scientific articles when scaled by 50%
WIDTH_TWOCOL = 14.16

#: dict: The name of the different dvas matplotlib style sheets
PLOT_STYLES = {'base': 'base.mplstyle',
               'nolatex': 'nolatex.mplstyle',
               'latex': 'latex.mplstyle'}

#: list[str]: The default file extensions to save the plots into.
PLOT_TYPES = ['.png', '.pdf']

#: bool: A flag to display the plots or not.
PLOT_SHOW = True

#: dict: matches the GDP units name to their (better) plot format
UNIT_LABELS = {'K': r'K$^{\circ}$',
               'm': r'm',
               'hPa': r'hPa',
               'percent': r'\%',
               'm s-1': r'm s$^{-1}$',
               'degree': r'$^{\circ}$',
               }

@log_func_call(plot_logger)
def set_mplstyle(style='base'):
    """ Set the DVAS plotting style. 'base' contains all the generic commands. 'latex'
    enables the use of a system-wide LaTeX engine. 'nolatex' disables it.

    Args:
        name (str or dict, optional): A style specification. Defaults to 'base'. Valid options are:

            - *str*: One of ['base', 'nolatex', 'latex'] defined inside DVAS.
            - *dict*: Dictionary with valid key/value pairs for matplotlib.rcParams.

    Caution:
        Specifying the 'latex' style requires a working system-wide LaTeX installation.
        Specifying 'nolatex' enables the default matplotlib LaTeX.

    Note:
        Users willing to override/enhance the default 'base' style can do so by feeding a dict of
        valid rcParams codes to this function.

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
    plt.style.use(str(Path(pkg_path, 'plot', 'mpl_styles', PLOT_STYLES['base'])))

    # Then apply which ever alternative style was requested, if we haven't already.
    if style != 'base':
        plt.style.use(str(Path(pkg_path, 'plot', 'mpl_styles', PLOT_STYLES[style])))

def cmap_discretize(cmap, n_cols):
    """Return a discrete colormap from the continuous colormap cmap.

    Args:
        cmap (str): colormap name or instance.
        n_cols (int): number of colors.

    Note:
        Adapted from the `Scipy Cookbook
        <https://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html>`__.

    Example:
        ::

            x = resize(arange(100), (5,100))
            djet = cmap_discretize(cm.jet, 5)
            imshow(x, cmap=djet)

    """

    # If I get a string, assume it's a proper colormap name
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    # Modification: do not start with the colormap edges
    #colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_i = np.concatenate((np.linspace(1./n_cols*0.5, 1-(1./n_cols*0.5), n_cols), (0., 0., 0.)))

    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., n_cols+1)

    cdict = {}
    for k_ind, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i],
                       colors_rgba[i-1, k_ind],
                       colors_rgba[i, k_ind]) for i in range(n_cols+1)]

    # Return colormap object.
    return colors.LinearSegmentedColormap(cmap.name + "_%d" % (n_cols), cdict, 1024)
