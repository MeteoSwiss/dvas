"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Plotting functions related to the gruan submodule.

"""

# Import from Python packages
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib import cm

# Import from this package
from ..dvas_logger import plot_logger, log_func_call
from . import plot_utils as pu
from ..dvas_environ import path_var


@log_func_call(plot_logger)
def pks_cmap(alpha=0.27/100, vmin=0.0, vmax=3*0.27/100):
    ''' Defines a custom colormap for the p-value plot of the KS test function.

    Args:
        alpha (float): the significance level of the KS test.
        vmin (float): vmin of the desired colorbar, for proper scaling of the transition level.
        vmax (float): vmax of the desired colorbar, for proper scaling of the transition level.

    Returns:
       matplotlib.colors.LinearSegmentedColormap

    '''

    # Some sanity checks
    if not isinstance(vmin, float) or not isinstance(vmax, float):
        raise Exception('Ouch ! vmin and vmax should be of type float, not %s and %s.' %
                        (type(vmin), type(vmax)))

    if not 0 <= vmin <= vmax <= 1:
        raise Exception('Ouch ! I need 0 <= vmin <= vmax <= 1.')


    # What are the boundary colors I want ?
    a_start = colors.to_rgb('maroon')
    a_mid_m = colors.to_rgb('lightcoral')
    a_mid_p = colors.to_rgb('lightgrey')
    a_end = colors.to_rgb('white')

    cdict = {}
    for c_ind, c_name in enumerate(['red', 'green', 'blue']):
        cdict[c_name] = ((0.00, a_start[c_ind], a_start[c_ind]),
                         ((alpha-vmin)/(vmax-vmin), a_mid_m[c_ind], a_mid_p[c_ind]),
                         (1.00, a_end[c_ind], a_end[c_ind])
                         )

    # Build the colormap
    return colors.LinearSegmentedColormap('pks_cmap', cdict, 1024)

@log_func_call(plot_logger)
def plot_ks_test(k_pqi, f_pqi, p_ksi, binning_list, alpha, tag=''):
    ''' Creates a diagnostic plot for the KS test.

    Args:
        k_pqi (ndarray): the **unbinned** normalized profile delta.
        f_pqi (ndarray): flags resulting from the rolling KS test.
        p_ksi (ndarray): p-values computed from the rolling KS test.
        binning_list (list of int): list of binning values used for the rolling KS test
        alpha (float): significance level used for the flags. Must be 0 <= alpha <= 1.
        tag (str): str to add to the filename for tagging.

    TODO:
        * fix the problem of thin lines becoming invisible (in case of long profiles)

    '''

    # Some sanity checks first
    if not isinstance(binning_list, list):
        raise Exception('Ouch ! binning_list must be a list of int, not %s' % (type(binning_list)))

    if not isinstance(alpha, float):
        raise Exception('Ouch ! alpha must be a float, and not %s' % (type(alpha)))
    if not 0 <= alpha <= 1:
        raise Exception('Ouch ! alpha=%.2f is invalid. Alpha must be >= 0 and <=1. %.2f' % (alpha))

    if len(k_pqi) != np.shape(f_pqi)[1] or len(k_pqi) != np.shape(p_ksi)[1]:
        raise Exception('Ouch ! something has a bad dimension.')

    # How many bins do I have ?
    n_bins = len(binning_list)

    # Set up the scene ...
    plt.close(57)
    plt.figure(57, figsize=(pu.WIDTH_TWOCOL, 5))

    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 0.02, 0.08],
                           left=0.08, right=0.98, bottom=0.12, top=0.97, wspace=0.02, hspace=0.08)

    ax0 = plt.subplot(gs[0, 0]) # An imshow plot of p_pq^i
    cax1 = plt.subplot(gs[0, 1]) # Color bar for p_pq^i
    ax1 = plt.subplot(gs[1, 0]) # A scatter plot of k_pq^i
    cax2 = plt.subplot(gs[1, 1]) # A histogram of k_pq^i

    # Let us begin with f_pq^i. Make the plot in % for a better readability.
    # TODO: for m=0, the lines are so thin that most are not actually visible in the imshow plot!
    ax0_map = ax0.imshow(100*p_ksi, cmap=pks_cmap(alpha=100*alpha, vmin=0., vmax=3.*100*alpha),
                         aspect='auto', interpolation='nearest',
                         vmin=0, vmax=3*100*alpha)

    # Add the colorbar
    cb1 = plt.colorbar(mappable=ax0_map, cax=cax1)
    cb1.set_label(r'$\left.p_{KS}^i\right|_{m}$ [\%]', labelpad=10)

    # Let's plot the unbinned delta.
    ax1.scatter(range(len(k_pqi)), k_pqi, marker='.', c='darkgrey')

    # Let us also clearly mark the flagged ones, coloring them as a function of the binning step
    # that found them.
    # First, I need an array of the time step indices for all the f_pqi values
    inds = np.tile(np.array([np.arange(0, len(k_pqi), 1)]), (n_bins, 1))
    # Let's also keep track of the associated binning.
    # Note that here I keep track of the binning "index" rather than the actual value.
    # That way, this does not blow up if the users specify non-contiguous values.
    bins = np.array([[bin_ind] * len(k_pqi) for bin_ind in range(n_bins)])

    # To draw circles (and not disks), I need to build the full color array to feed to edgecolors.
    # See: https://stackoverflow.com/questions/43519160/
    # matplotlib-scatter-plot-with-colormaps-for-edgecolor-but-no-facecolor
    # and the reply from ImportanceOfBeingEarnest
    # Let's not forget to rescale the values as well !
    bin_cm = pu.cmap_discretize('plasma_r', n_bins) # That's the colormap I want, only discretized.
    # Turn binning values into colors. We need to map -0.5 <-> n_bins-0.5 onto 0 <-> 1.
    # That's where the 0.5 comes from. This will let us align the major ticks with the colormap
    # bins.
    flag_origs = bin_cm((bins[f_pqi == 1] + 0.5)/n_bins)

    # Now actually plot circles around the flagged values.
    ax1.scatter(inds[f_pqi == 1], np.tile(k_pqi, (n_bins, 1))[f_pqi == 1],
                marker='o',
                s=5*(bins[f_pqi == 1]+1)**2, # With this, we get circles growing linearly in radius
                edgecolors=flag_origs, # We color each circle manually. No cmap !!!
                linewidth=1, facecolor='none')

    # Add the 0 line for reference.
    ax1.axhline(0, c='k', ls='--', lw=1)

    # Add the discretrized colorbar and ticks.
    # Since I did not use 'cmap' in the scatter function, I must draw the colorbar manually.
    # https://matplotlib.org/tutorials/colors/colorbar_only.html?highlight=colorbarbase
    # Note how the vmin & vmax value match what we did earlier.
    cb2 = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-0.5, vmax=n_bins-0.5),
                                         cmap=bin_cm),
                       cax=cax2)

    # Back to solid ground. Let's label things properly.
    cb2.set_label(r'$m$', labelpad=10)
    # For the ticks, let's tag them with the actual binning value.
    # Downside: this will get very crowded for lots of binning.
    cb2.set_ticks(range(n_bins))
    cb2.set_ticklabels([r'%i' % (m_val) for m_val in binning_list])
    cb2.ax.tick_params(which='minor', axis='y', left=False, right=False)

    # Set the proper axis labels, etc ...
    for ax in [ax0, ax1]:
        ax.set_xlim((-0.5, len(k_pqi)-0.5))

    ax0.set_ylabel(r'$m$')
    ax1.set_ylabel(r'$k_{p,q}^i$')
    ax1.set_xlabel(r'$i$')

    ax0.set_ylim((-0.5 + n_bins, -0.5))
    ax0.set_yticks(np.arange(0, n_bins, 1))
    ax0.set_yticklabels([r'%i' % (m_val) for m_val in binning_list])
    ax0.set_xticklabels([])
    ax0.tick_params(which='minor', axis='y', left=False, right=False)

    ax1.set_ylim((-6, 6))

    # Save the figure
    plt.savefig(path_var.plot_output_path / ('k_pqi_%s.pdf' % (tag)))
    plt.show()
