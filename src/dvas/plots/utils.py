"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

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
from cycler import cycler

# Import from this package
from ..hardcoded import PKG_PATH
from ..errors import DvasError
from ..environ import path_var as env_path_var
from ..logger import log_func_call
from ..logger import plots_logger as logger
from ..version import VERSION

# Define some plotting constants
#: float: Width of a 1-column plot [inches], to fit in scientific articles when scaled by 50%
WIDTH_ONECOL = 6.92

#: float: Width of a 2-column plot [inches], to fit in scientific articles when scaled by 50%
WIDTH_TWOCOL = 14.16

#: dict: The name of the different dvas matplotlib style sheets
PLOT_STYLES = {'base': 'base.mplstyle',
               'nolatex': 'nolatex.mplstyle',
               'latex': 'latex.mplstyle'}

#: dict: dvas core colors for the cmap, the color cycler, and NaNs.
CLRS = {'cmap_anchors': ['#351659', '#67165b', '#901f55', '#af374a', '#c3563e', '#cc7a33',
                         '#c38e49', '#bc9f64', '#b9ae82', '#babaa2', '#c3c4c0'],
        'nan': '#7d7d7d',
        }

#: matplotlib.colors.LinearSegmentedColormap: the default dvas colormap
DVAS_CMAP_1 = colors.LinearSegmentedColormap.from_list('dvas_cmap_1', CLRS['cmap_anchors'], 1024)
DVAS_CMAP_1_r = colors.LinearSegmentedColormap.from_list('dvas_cmap_1', CLRS['cmap_anchors'][::-1],
                                                         1024)


#: list[str]: The default file extensions to save the plots into.
PLOT_FMTS = ['png']

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

@log_func_call(logger)
def set_mplstyle(style='base'):
    """ Set the DVAS plotting style. 'base' contains all the generic commands. 'latex'
    enables the use of a system-wide LaTeX engine. 'nolatex' disables it.

    Args:
        name (str | dict, optional): A style specification. Defaults to 'base'. Valid options are:

            - *str*: One of ['base', 'nolatex', 'latex'] defined inside DVAS.
            - *dict*: Dictionary with valid key/value pairs for matplotlib.rcParams.

    Caution:
        Specifying the 'latex' style requires a working system-wide LaTeX installation.
        Specifying 'nolatex' enables the default matplotlib LaTeX.

    Note:
        Users willing to override/enhance the default 'base' style can do so by feeding a dict of
        valid rcParams codes to this function.

    """

    # Always apply the base style first.
    plt.style.use(str(Path(PKG_PATH, 'plots', 'mpl_styles', PLOT_STYLES['base'])))

    # Update the color cycler to match our custom colorscheme
    n_anchors = len(CLRS['cmap_anchors'])
    default_cycler = (cycler(color=[CLRS['cmap_anchors'][(3*ind) % n_anchors]
                                    for ind in range(10)]))
    plt.rc('axes', prop_cycle=default_cycler)

    # Let's start with some sanity checks. If the user is foolish enough to feed a dict, trust it.
    if isinstance(style, dict):
        plt.style.use(style)

    # Else, if this is not a known str, let's be unforgiving.
    if not isinstance(style, str):
        raise Exception('Ouch ! style type must be one of (str, dict), not: %s' % (type(style)))

    if style not in PLOT_STYLES.keys():
        raise Exception('Ouch! plot style "%s" unknown. Should be one of [%s].'
                        % (style, ', '.join(PLOT_STYLES.keys())))

    # Then apply which ever alternative style was requested, if we haven't already.
    if style != 'base':
        plt.style.use(str(Path(PKG_PATH, 'plots', 'mpl_styles', PLOT_STYLES[style])))


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

@log_func_call(logger)
def fancy_legend(ax, label=None):
    """ A custom legend routine, to take care of all the repetitive aspects for this.

    Args:
        ax (matplotlib.pyplot.axes): the plot axes to add the legend to.
        label (str, optional): the legend label

    """

    # If I am using some fancy LaTeX, let's make sue that I escape all the nasty characters.
    if plt.rcParams['text.usetex']:
        label = label.replace('_', '\_')

    # Add the legend.
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                    title=label, ncol=1, handlelength=1, fancybox=True,
                    fontsize='small', title_fontsize='small', borderaxespad=0)

    # Tweak the thickness of the legen lines as well.
    for line in leg.get_lines():
        line.set_linewidth(2.0)

@log_func_call(logger)
def add_edt_eid_rid(ax, prfs):
    """ Add basic edt, eid and rid info to a plot.

    Args:
        ax (matplotlib.pyplot.axes): the axes to add the info to.
        prfs (Multiprofile|MultiRSProfile|MultiGDPProfile): the MultiProfile to extract the info
            from.

    """

    # Get the data to print
    edts = [item.strftime('%Y-%m-%d %H:%M %Z') for item in prfs.get_info('edt')]
    eids = prfs.get_info('eid')
    rids = prfs.get_info('rid')

    # Format it all nicely for each Profile.
    info_txt = set(['{} ({}, {})'.format(item, eids[ind], rids[ind])
                                         for ind, item in enumerate(edts)])
    # Make sure it does not overflow
    info_txt = '\n'.join(info_txt)

    # Add it to the ax
    ax.text(0.98, 0.95, info_txt, fontsize='small',
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes)

@log_func_call(logger)
def add_source(fig):
    """ Add a sentence about the dvas version to a given plot.

    Args:
        fig (matplotlib.pyplot.figure): the figure to add the text to.

    """
    msg = 'Created with dvas v{}'.format(VERSION)

    fig.text(0.99, 0.02, msg, fontsize='xx-small',
                 horizontalalignment='right', verticalalignment='bottom')

@log_func_call(logger)
def fancy_savefig(fig, fn_core, fn_prefix=None, fn_suffix=None, fmts=None, show=None):
    """ A custom savefig function that provides finer handling of the filename.

    Args:
        fig (matplotlib.figure): the figure to save.
        fn_core (str): the core part of the filename.
        fn_prefix (str, optional): a prefix, to which fn_core will be appended with a '_'.
            Defauts to None.
        fn_suffix (str, optional): a suffix, that will be appended to fn_core with a '_'.
        fmts (str|list of str, optional): which formats to export the plot to, e.g.: 'png'.
            Defaults to None (= as specified by dvas.plots.utils.PLOT_FMTS)
        show (bool, optional): whether to display the plot after saving it, or not. Defaults to
            None (= as specified by dvas.plots.utils.PLOT_SHOW)
    """

    # Same sanity checks first. If the fmt is a str, turn it into a list.
    if fmts is None:
        fmts = PLOT_FMTS
    if isinstance(fmts, str):
        fmts = [fmts]

    if show is None:
        show = PLOT_SHOW

    # Build the fileneame
    fn_out = '_'.join([item for item in [fn_prefix, fn_core, fn_suffix] if item is not None])

    # Let us first make sure the destination folder has been set ...
    if env_path_var.output_path is None:
        raise DvasError('Ouch ! dvas.environ.path_var.output_path is None')
    # ... and that the location exists.
    if not env_path_var.output_path.exists():
        # If not, be bold and create the folder.
        env_path_var.output_path.mkdir(parents=True)
        # Set user read/write permission
        env_path_var.output_path.chmod(env_path_var.output_path.stat().st_mode | 0o600)

    # Save the figure in all the requested formats
    for fmt in fmts:

        # Make sure I can actually deal with the format ...
        if fmt not in fig.canvas.get_supported_filetypes().keys():
            logger.warning('%s format not supported by the OS. Ignoring it.', fmt)
            continue

        # Note: never use a tight box fix here. If the plot looks weird, the gridspec
        # params should be altered. This is essential for the consistency of the DVAS plots.
        fig.savefig(Path(env_path_var.output_path, '.'.join([fn_out, fmt])))

    # Show the plot, or just close it and move on
    if show:
        fig.show()
    else:
        plt.close(fig.number)

#@log_func_call(logger)
#def pks_cmap(alpha=0.27/100, vmin=0.0, vmax=3*0.27/100):
#    """ Defines a custom colormap for the p-value plot of the KS test function.
#
#    Args:
#        alpha (float): the significance level of the KS test.
#        vmin (float): vmin of the desired colorbar, for proper scaling of the transition level.
#        vmax (float): vmax of the desired colorbar, for proper scaling of the transition level.
#
#    Returns:
#       matplotlib.colors.LinearSegmentedColormap
#
#    """
#
#    # Some sanity checks
#    if not isinstance(vmin, float) or not isinstance(vmax, float):
#        raise Exception('Ouch ! vmin and vmax should be of type float, not %s and %s.' %
#                        (type(vmin), type(vmax)))
#
#    if not 0 <= vmin <= vmax <= 1:
#        raise Exception('Ouch ! I need 0 <= vmin <= vmax <= 1.')
#
#
#    # What are the boundary colors I want ?
#    a_start = colors.to_rgb('maroon')
#    a_mid_m = colors.to_rgb('lightcoral')
#    a_mid_p = colors.to_rgb('lightgrey')
#    a_end = colors.to_rgb('white')
#
#    cdict = {}
#    for c_ind, c_name in enumerate(['red', 'green', 'blue']):
#        cdict[c_name] = ((0.00, a_start[c_ind], a_start[c_ind]),
#                         ((alpha-vmin)/(vmax-vmin), a_mid_m[c_ind], a_mid_p[c_ind]),
#                         (1.00, a_end[c_ind], a_end[c_ind])
#                         )
#
#    # Build the colormap
#    return colors.LinearSegmentedColormap('pks_cmap', cdict, 1024)
