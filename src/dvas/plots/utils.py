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
import matplotlib.gridspec as gridspec
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
CLRS = {'set_1': ['#4c88b3', '#b34c88', '#88b34c', '#d15f56', '#7d70b4', '#00a7a0',
                  '#bf8a31', '#d0d424','#b3b3b3','#575757'],
        'cmap_1': list(zip([0, 0, 0.5, 1, 1],
                           ['#000000', '#051729','#4c88b3','#dbd7cc', '#ffffff'])),
        'nan_1': '#7d7d7d',
        }

#: matplotlib.colors.LinearSegmentedColormap: the default dvas colormap 1
CMAP_1 = colors.LinearSegmentedColormap.from_list('cmap_1', CLRS['cmap_1'], 1024)
CMAP_1.set_bad(color=CLRS['nan_1'], alpha=1)
#: matplotlib.colors.LinearSegmentedColormap: the default dvas colormap 1 reversed
CMAP_1_R = colors.LinearSegmentedColormap.from_list('cmap_1',
    [(abs(item[0]-1), item[1]) for item in CLRS['cmap_1'][::-1]], 1024)
CMAP_1_R.set_bad(color=CLRS['nan_1'], alpha=1)

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
    n_clrs = len(CLRS['set_1'])
    default_cycler = (cycler(color=CLRS['set_1']))
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

def clr_palette_demo():
    """ A simple function to demonstrate the dvas color palette.

    """

    plt.close(99)
    fig = plt.figure(99, figsize=(WIDTH_TWOCOL, 4.5))

    # Use gridspec for a fine control of the figure area.
    fig_gs = gridspec.GridSpec(1, 6, height_ratios=[1],
                               width_ratios=[20, 10, 1, 1, 1, 1],
                               left=0.05, right=0.97, bottom=0.05, top=0.95,
                               wspace=0.05, hspace=0.05)

    # Instantiate the axes
    ax0a = fig.add_subplot(fig_gs[0, 0])
    ax0b = fig.add_subplot(fig_gs[0, 1])
    ax1 = fig.add_subplot(fig_gs[0, 2])
    ax2 = fig.add_subplot(fig_gs[0, 3])
    ax3 = fig.add_subplot(fig_gs[0, 4])
    ax4 = fig.add_subplot(fig_gs[0, 5])

    # First, plot some lines to illustrate the set colors ...
    for (ind, clr) in enumerate(CLRS['set_1']):
        ax0a.plot(np.sin(np.linspace(0, 2*np.pi, 100) + 2*np.pi*(1-ind/len(CLRS['set_1']))),
                  '-', c=clr, label='{}'.format(clr))
        ax0a.fill_between(np.arange(50, 100, 1),
                          np.sin(np.linspace(np.pi, 2*np.pi, 50) +
                                 2*np.pi*(1-ind/len(CLRS['set_1'])))-0.1,
                          np.sin(np.linspace(np.pi, 2*np.pi, 50) +
                                 2*np.pi*(1-ind/len(CLRS['set_1'])))+0.1,
                          '-', facecolor=clr, alpha=0.4, edgecolor=None)

    # Show the color names as legend
    ax0a.legend(fontsize='xx-small')

    # Second, show a 2D image with the defasult colormap ...
    x, y = np.meshgrid(np.linspace(-3,3,30), np.linspace(-3,3,30))
    d = np.sqrt(x**2 + y**2)
    g = np.exp(-( (d)**2 / ( 2.0 * 1**2 ))) + (2*np.random.rand(30,30)-1)/10
    # Add some NaN's to show their specific color
    g[0:15,0:15] = np.nan

    # Actually plot it ...
    ax0b.imshow(g, cmap=CMAP_1, origin='lower', vmin=0, vmax=1)
    ax0b.text(7.5, 7.5, 'NaN', horizontalalignment='center', verticalalignment='center')

    # Clean it up a bit ...
    for ax in [ax0b, ax0a]:
        ax.tick_params(which='both', length=0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # Then, show an actual colorbar in full ...
    plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1),
                                   cmap=CMAP_1), cax=ax1)
    # ... then binned in 3 chunks ...
    plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1),
                                   cmap=cmap_discretize(CMAP_1, 3)), cax=ax2)
    # ... and 5 chunks ...
    plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1),
                                   cmap=cmap_discretize(CMAP_1, 5)), cax=ax3)
    # ... and 10 chunks ...
    plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1),
                                   cmap=cmap_discretize(CMAP_1, 10)), cax=ax4)

    # Clean up the ticks ...
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)


    # Show and save
    plt.show()
