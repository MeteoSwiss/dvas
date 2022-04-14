"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Plotting functions related to the gruan submodule.

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
def gdps_vs_cws(gdp_prfs, cws_prf, k_lvl=1, label='mid', **kwargs):
    """ Makes a plot comparing different GDPs with their associated combined working measurement
    standard.

    All profiles must imperatively be fully synchronized.

    Args:
        gdp_prfs (dvas.data.data.MultiGDPProfile): the GDPs
        cws_prf (dvas.data.data.MultiCWSProfile): the combined working standards, for example
            generated by dvas.tools.gruan.combine_gdps().
        k_lvl (int|float, optional): k-level for the uncertainty. Defaults to 1.
        label (str, optional): label of the plot legend. Defaults to 'mid'.
        **kwargs: these get fed to the dvas.plots.utils.fancy_savefig() routine.

    Returns:
        matplotlib.pyplot.figure: the figure instance
    """

    # Start the plotting
    fig = plt.figure(figsize=(pu.WIDTH_TWOCOL, 6.5))

    # Create a gridspec structure
    gs_info = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[1],
                                left=0.09, right=0.87, bottom=0.23, top=0.93,
                                wspace=0.5, hspace=0.1)

    # Create the axes - one for the profiles, and one for uctot, ucr, ucs, uct, ucu
    ax0 = fig.add_subplot(gs_info[0, 0])
    ax1 = fig.add_subplot(gs_info[1, 0], sharex=ax0)

    # Extract the DataFrames from the MultiGDPProfile instances
    cws = cws_prf.get_prms([PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME,
                            PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME,
                            'uc_tot'])[0]
    gdps = gdp_prfs.get_prms([PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME,
                             PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME,
                             'uc_tot'])

    # Let us make sure that all the profiles are synchronized by checking the profile lengths
    # This is not fool proof, but it is a start. I could also check for the sync tag, but it
    # would be less strict as a check.
    if len(cws) != len(gdps):
        raise DvasError('Ouch! GDPS and CWS do not have the same lengths. I cannot plot this.')

    # Here, I want to make a plot with 3 x-axis: idx, tdt, and alt. To do that requires
    # all these arrays to be completely filled with stuff, an dhave absolutely no NaNs
    # (since matplotlib needs to interpolate and go back-and-forth between them)
    idxs = cws.index.get_level_values(PRF_REF_INDEX_NAME).values
    tdts = cws[PRF_REF_TDT_NAME].dt.total_seconds().values
    alts = cws[PRF_REF_ALT_NAME].values

    # For the altitude, I may need to pad the edges with stuff. Holes, on the other hand, should
    # already have been plugged.
    min_valid_alt_idx = cws.index[~cws['alt'].isna()].min()
    max_valid_alt_idx = cws.index[~cws['alt'].isna()].max()
    alts[:min_valid_alt_idx] = np.arange(0, min_valid_alt_idx, 1) - 1e6 + 1
    alts[max_valid_alt_idx+1:] = np.arange(max_valid_alt_idx+1-len(alts), 0, 1) + 1e6

    # Check for any holes - if found, forget about showing multiple axes
    show_tdts = True
    if np.any(np.isnan(tdts)):
        logger.error('CWS tdt contains holes. ' +
                     'Axis will be cropped from gdp_vs_cws.')
        show_tdts = False
    # Make sure time is monotically increasing. This should always be true ...
    elif not np.all(np.diff(tdts) > 0):
        raise DvasError('Ouch ! CWS tdts not increasing monotically ?!')

    # For the altitude, things may not be that easy. Ballons can go down temporarily in gravity
    # waves, for exemple. Do circumvent this, we shall display the alt axis only if it increases
    # monotically
    show_alts = True
    if np.any(np.isnan(alts)):
        logger.error('CWS ref_alt contains holes. ' +
                     'Axis will be cropped from gdp_vs_cws.')
        show_alts = False
    elif np.any(np.diff(alts) <= 0):
        logger.error('CWS ref. alt. is not increasing monotically. ' +
                     'Axis will be cropped from gdp_vs_cws')
        show_alts = False

    # Very well, let us plot all these things.
    for gdp_ind in range(len(gdps.columns.levels[0])):

        gdp = gdps[gdp_ind]

        # First, plot the profiles themselves
        ax0.plot(idxs, gdp[PRF_REF_VAL_NAME].values, lw=0.5, ls='-', drawstyle='steps-mid',
                 label='|'.join(gdp_prfs.get_info(label)[gdp_ind]))
        ax0.fill_between(idxs, gdp[PRF_REF_VAL_NAME].values-k_lvl*gdp['uc_tot'].values,
                         gdp[PRF_REF_VAL_NAME]+k_lvl*gdp['uc_tot'], alpha=0.3, step='mid')

        # Then, plot the Deltas with respect to the CWS
        delta = gdp[PRF_REF_VAL_NAME].values-cws[PRF_REF_VAL_NAME].values
        ax1.plot(idxs, delta, drawstyle='steps-mid', lw=0.5, ls='-')
        ax1.fill_between(idxs, delta-k_lvl*gdp['uc_tot'], delta+k_lvl*gdp['uc_tot'], alpha=0.3,
                         step='mid')

    # Then also plot the CWS uncertainty
    ax0.plot(idxs, cws['val'], color=pu.CLRS['cws_1'], lw=0.5, ls='-', drawstyle='steps-mid',
             label='CWS')
    ax0.fill_between(idxs, cws[PRF_REF_VAL_NAME]-k_lvl*cws['uc_tot'],
                     cws[PRF_REF_VAL_NAME]+k_lvl*cws['uc_tot'], alpha=0.3, step='mid',
                     color=pu.CLRS['cws_1'])

    ax1.plot(idxs, -k_lvl*cws['uc_tot'].values, lw=0.5, drawstyle='steps-mid',
             color='k')
    ax1.plot(idxs, +k_lvl*cws['uc_tot'].values, lw=0.5, drawstyle='steps-mid',
             color='k')

    # Make it look pretty
    ylbl = cws_prf.var_info[PRF_REF_VAL_NAME]['prm_name']
    ylbl += ' [{}]'.format(cws_prf.var_info[PRF_REF_VAL_NAME]['prm_unit'])

    idxlbl = r'$i$'
    tdtlbl = cws_prf.var_info[PRF_REF_TDT_NAME]['prm_name']
    tdtlbl += ' [{}]'.format(cws_prf.var_info[PRF_REF_TDT_NAME]['prm_unit'])
    altlbl = cws_prf.var_info[PRF_REF_ALT_NAME]['prm_name'] + r'$_{\rm CWS}$'
    altlbl += ' [{}]'.format(cws_prf.var_info[PRF_REF_ALT_NAME]['prm_unit'])

    # idx axis
    ax1.text(1.02, -0.11, pu.fix_txt(idxlbl), ha='left', va='center', transform=ax1.transAxes)

    # Now deal with the multiple axis I need to show.
    if show_tdts:
        tdtax = pu.add_sec_axis(ax1, idxs, tdts, offset=-0.25, which='x')
        tdtax.tick_params(labelsize='small')
        ax1.text(1.02, -0.34, pu.fix_txt(tdtlbl), ha='left', va='center', transform=ax1.transAxes,
                 fontsize='small')

    if show_alts:
        altax = pu.add_sec_axis(ax1, idxs, alts, offset=-0.5, which='x')
        altax.tick_params(labelsize='small')
        ax1.text(1.02, -0.59, pu.fix_txt(altlbl), ha='left', va='center', transform=ax1.transAxes,
                 fontsize='small')

    # Legends, labels, etc ...
    ax0.text(-0.1, 0.5, pu.fix_txt(ylbl), ha='left', va='center',
             transform=ax0.transAxes, rotation=90)
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax1.text(-0.1, 0.5, pu.fix_txt(r'$\Delta$' + ylbl), ha='left', va='center',
             transform=ax1.transAxes, rotation=90)

    # Crop the plot to the regions with valid ref_altitudes
    ax0.set_xlim((min_valid_alt_idx, max_valid_alt_idx))

    # Add the legend
    pu.fancy_legend(ax0, label)

    # Add the edt/eid/rid info
    pu.add_edt_eid_rid(ax0, cws_prf)

    # Add the k-level
    pu.add_var_and_k(ax0, var_name=cws_prf.var_info[PRF_REF_VAL_NAME]['prm_name'], k=k_lvl)

    # Add the source for the plot
    pu.add_source(fig)

    # Save it
    pu.fancy_savefig(fig, fn_core='gdps-vs-cws', **kwargs)


@log_func_call(logger)
def uc_budget(gdp_prfs, cws_prf, k_lvl=1, label='mid', **kwargs):
    """ Makes a plot of the uncertainty budget of GDPs and (as an option) associated CWS.

    All profiles must imperatively be fully synchronized. Don't try fancy things ...

    Args:
        gdp_prfs (dvas.data.data.MultiGDPProfile): the GDPs
        cws_prf (dvas.data.data.MultiCWSProfile): the combined working standards,
            for example generated by dvas.tools.gruan.combine_gdps(). Defaults to None.
        k_lvl (int|float, optional): k-level for the uncertainty. Defaults to 1.
        label (str, optional): label of the plot legend. Defaults to 'mid'.
        **kwargs: these get fed to the dvas.plots.utils.fancy_savefig() routine.

    Returns:
        matplotlib.pyplot.figure: the figure instance
    """

    # Start the plotting
    fig = plt.figure(figsize=(pu.WIDTH_TWOCOL, 9.5))

    # Create a gridspec structure
    gs_info = gridspec.GridSpec(6, 1, height_ratios=[1]*6, width_ratios=[1],
                                left=0.09, right=0.87, bottom=0.15, top=0.95,
                                wspace=0.5, hspace=0.1)

    # Create the axes - one for the profiles, and one for uctot, ucr, ucs, uct, ucu
    ax0 = fig.add_subplot(gs_info[0, 0])
    ax0b = fig.add_subplot(gs_info[1, 0], sharex=ax0)
    ax1 = fig.add_subplot(gs_info[2, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs_info[3, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs_info[4, 0], sharex=ax0)
    ax4 = fig.add_subplot(gs_info[5, 0], sharex=ax0)
    # Keep a list to loop efficiently
    axs = [ax0, ax1, ax2, ax3, ax4]

    # Extract the DataFrames from the MultiGDPProfile/MultiCWSProfile instances
    gdps = gdp_prfs.get_prms([PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME,
                             PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME,
                             'uc_tot'])

    cws = cws_prf.get_prms([PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME,
                            PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME,
                            PRF_REF_UCU_NAME, 'uc_tot'])[0]

    # Let us make sure that all the profiles are synchronized by checking the profile lengths
    # This is not fool proof, but it is a start. I could also check for the sync tag, but it
    # would be less strict as a check.
    if len(cws) != len(gdps):
        raise DvasError('Ouch! GDPS and CWS do not have the same lengths. I cannot plot this.')

    # Here, I want to make a plot with 3 x-axis: idx, tdt, and alt. To do that requires
    # all these arrays to be completely filled with stuff, and have absolutely no NaNs
    # (since matplotlib needs to interpolate and go back-and-forth between them)
    idxs = cws.index.get_level_values(PRF_REF_INDEX_NAME).values
    tdts = cws[PRF_REF_TDT_NAME].dt.total_seconds().values
    alts = cws[PRF_REF_ALT_NAME].values

    # For the altitude, I may need to pad the edges with stuff. Holes, on the other hand, should
    # already have been plugged.
    min_valid_alt_idx = cws.index[~cws['alt'].isna()].min()
    max_valid_alt_idx = cws.index[~cws['alt'].isna()].max()
    alts[:min_valid_alt_idx] = np.arange(0, min_valid_alt_idx, 1) - 1e6 + 1
    alts[max_valid_alt_idx+1:] = np.arange(max_valid_alt_idx+1-len(alts), 0, 1) + 1e6

    # Check for any holes - if found, forget about showing multiple axes
    show_tdts = True
    if np.any(np.isnan(tdts)):
        logger.error('CWS tdt contains holes. ' +
                     'Axis will be cropped from uc_budget.')
        show_tdts = False
    # Make sure time is monotically increasing. This should always be true ...
    elif not np.all(np.diff(tdts) > 0):
        raise DvasError('Ouch ! CWS tdts not increasing monotically ?!')

    # For the altitude, things may not be that easy. Ballons can go down temporarily in gravity
    # waves, for exemple. Do circumvent this, we shall display the alt axis only if it increases
    # monotically
    show_alts = True
    if np.any(np.isnan(alts)):
        logger.error('CWS ref_alt contains holes. ' +
                     'Axis will be cropped from uc_budget.')
        show_alts = False
    elif np.any(np.diff(alts) <= 0):
        logger.error('CWS ref. alt. is not increasing monotically. ' +
                     'Axis will be cropped from uc_budget.')
        show_alts = False

    # Very well, let us plot all these things. First the GDPs ...
    for gdp_ind in range(len(gdps.columns.levels[0])):
        gdp = gdps[gdp_ind]

        # Finally, plot the individual errors too ...
        for (uc_ind, uc) in enumerate(['uc_tot', PRF_REF_UCR_NAME, PRF_REF_UCS_NAME,
                                       PRF_REF_UCT_NAME, PRF_REF_UCU_NAME]):
            axs[uc_ind].plot(idxs, k_lvl*gdp[uc].values, drawstyle='steps-mid', lw=0.5,
                             label='|'.join(gdp_prfs.get_info(label)[gdp_ind]))

        # Let's also plot the weights (from the GDP combination scheme)
        ax0b.plot(idxs, 1/gdp['uc_tot'].values**2, lw=0.5, drawstyle='steps-mid')

    # Then also plot the CWS uncertainty
    for (uc_ind, uc) in enumerate(['uc_tot', PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME,
                                   PRF_REF_UCU_NAME]):
        axs[uc_ind].plot(idxs, k_lvl*cws[uc].values, drawstyle='steps-mid', lw=0.75, c='k',
                         zorder=0, label='CWS')
        # Add the y-label, while I'm at it ...
        axs[uc_ind].text(-0.1, 0.5, pu.fix_txt(uc), ha='left', va='center',
                         transform=axs[uc_ind].transAxes, rotation=90)
        # Set the lower limits as well
        axs[uc_ind].set_ylim(bottom=0)

    # Let's also plot the weights (from the GDP combination scheme)
    ax0b.set_ylim(bottom=0)
    ax0b.text(-0.1, 0.5, pu.fix_txt('1/uc_tot$^2$'), ha='left', va='center',
              transform=ax0b.transAxes, rotation=90)

    # Now make it look pretty
    idxlbl = r'$i$'
    tdtlbl = cws_prf.var_info[PRF_REF_TDT_NAME]['prm_name']
    tdtlbl += ' [{}]'.format(cws_prf.var_info[PRF_REF_TDT_NAME]['prm_unit'])
    altlbl = cws_prf.var_info[PRF_REF_ALT_NAME]['prm_name'] + r'$_{\rm CWS}$'
    altlbl += ' [{}]'.format(cws_prf.var_info[PRF_REF_ALT_NAME]['prm_unit'])

    # idx axis
    ax4.text(1.02, -0.15, pu.fix_txt(idxlbl), ha='left', va='center', transform=ax4.transAxes)

    # Now deal with the multiple axis I need to show.
    if show_tdts:
        tdtax = pu.add_sec_axis(ax4, idxs, tdts, offset=-0.4, which='x')
        tdtax.tick_params(labelsize='small')
        ax4.text(1.02, -0.5, pu.fix_txt(tdtlbl), ha='left', va='center', transform=ax4.transAxes,
                 fontsize='small')

    if show_alts:
        altax = pu.add_sec_axis(ax4, idxs, alts, offset=-0.8, which='x')
        altax.tick_params(labelsize='small')
        ax4.text(1.02, -0.9, pu.fix_txt(altlbl), ha='left', va='center', transform=ax4.transAxes,
                 fontsize='small')

    # Legends, labels, etc ...
    for ax in [ax0, ax0b, ax1, ax2, ax3]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # Crop the plot to the regions with valid ref_altitudes
    ax0.set_xlim((min_valid_alt_idx, max_valid_alt_idx))

    # Add the legend
    pu.fancy_legend(ax0, label)
    # Add the k-level
    var_msg = r'{} [{}]'.format(cws_prf.var_info[PRF_REF_VAL_NAME]['prm_name'],
                                cws_prf.var_info[PRF_REF_VAL_NAME]['prm_unit'])
    pu.add_var_and_k(ax0, var_name=var_msg, k=k_lvl)

    # Add the edt/eid/rid info
    pu.add_edt_eid_rid(ax0, cws_prf)

    # Add the source for the plot
    pu.add_source(fig)

    # Save it
    pu.fancy_savefig(fig, fn_core='uc_budget', **kwargs)


@log_func_call(logger)
def plot_ks_test(df, alpha, unit=None, left_label=None, right_label=None, **kwargs):
    """ Creates a diagnostic plot for the KS test.

    Args:
        df (pd.DataFrame): a very special DataFrame, generated inside
            dvas.tools.gdps.stats.get_incomptibility().
        alpha (float): significance level used for the flags. Must be 0 <= alpha <= 1.
            Required for setting up the colorbar properly.
        unit (str, optional): the unit of the variable displayed. Defaults to None.
        left_label (str, optional): top-left plot label. Defaults to None.
        right_label (str, optional): top-right plot label. Defaults to None.
        **kwargs: these get fed to the dvas.plots.utils.fancy_savefig() routine.

    Returns:
        matplotlib.pyplot.figure: the figure instance
    """

    # Some sanity checks first
    if not isinstance(alpha, float):
        raise DvasError('Ouch ! alpha must be a float, and not %s' % (type(alpha)))
    if not 0 <= alpha <= 1:
        raise DvasError('Ouch ! alpha={} is invalid. Should be >= 0 and <=1.'.format(alpha))

    # How many different binnings do I have ?
    # Note: bin "0" contains the full-resolution delta, and does not count as a used binning for the
    # rolling KS test.
    n_bins = len(df.columns.levels[0])-1

    # The plot will have different number of rows depending on the number of variables.
    # Let's define some hardcoded heights, such that the look is always consistent
    top_gap = 0.4  # inch
    bottom_gap = 0.7  # inch
    plot_1_height = 0.35*n_bins  # inch
    plot_2_height = 2.  # inch
    mid_gap = 0.05  # inch
    fig_height = top_gap + bottom_gap + plot_1_height + 3*plot_2_height + 3*mid_gap

    # Set up the scene ...
    fig = plt.figure(figsize=(pu.WIDTH_TWOCOL, fig_height))

    gs_info = gridspec.GridSpec(4, 1, height_ratios=[plot_1_height/plot_2_height, 1, 1, 1],
                                width_ratios=[1], left=0.08, right=0.98,
                                bottom=bottom_gap/fig_height, top=1-top_gap/fig_height,
                                wspace=0.02, hspace=mid_gap)

    ax1 = fig.add_subplot(gs_info[0, 0])  # A 2D plot of the incompatible points as a function of m
    ax2 = fig.add_subplot(gs_info[1, 0], sharex=ax1)  # A scatter plot of k_pq^ei
    ax3 = fig.add_subplot(gs_info[2, 0], sharex=ax1)  # A scatter plot of Delta_pq^ei
    ax4 = fig.add_subplot(gs_info[3, 0], sharex=ax1)  # A scatter plot of sigma_pq^ei

    # First, let's plot the full-resolution data.
    ax2.scatter(df.index.values, df.loc[:, (0, 'k_pqei')], marker='o',
                facecolor=pu.CLRS['nan_1'], s=1, edgecolor=None, zorder=10)
    ax3.scatter(df.index.values, df.loc[:, (0, 'Delta_pqei')], marker='o',
                facecolor=pu.CLRS['nan_1'], s=1, edgecolor=None, zorder=10)
    ax4.scatter(df.index.values, df.loc[:, (0, 'sigma_pqei')], marker='o',
                facecolor=pu.CLRS['nan_1'], s=1, edgecolor=None, zorder=10)

    # For k, show the k=1, 2, 3 zones
    for k in [1, 2, 3]:
        ax2.fill_between([df.index.values[0], df.index.values[-1]], [-k, -k], [k, k],
                         alpha=0.1+(3-k)*0.1,
                         facecolor='mediumpurple', edgecolor='none')

    # Let's now deal with all the bin levels ...
    for bin_ind in range(n_bins):

        # Which levels have been flagged ?
        flagged = df[(df.columns.levels[0][1+bin_ind], 'f_pqei')] == 1

        # Plot the binned delta profile.
        ax2.plot(df.index.values, df.loc[:, (df.columns.levels[0][1+bin_ind], 'k_pqei')],
                 ls='-', color=pu.CLRS['nan_1'], lw=0.5, drawstyle='steps-mid')
        ax3.plot(df.index.values, df.loc[:, (df.columns.levels[0][1+bin_ind], 'Delta_pqei')],
                 ls='-', color=pu.CLRS['nan_1'], lw=0.5, drawstyle='steps-mid')
        ax4.plot(df.index.values, df.loc[:, (df.columns.levels[0][1+bin_ind], 'sigma_pqei')],
                 ls='-', color=pu.CLRS['nan_1'], lw=0.5, drawstyle='steps-mid')

        # Clearly mark the bad regions in the top plot
        ax1.errorbar(df[flagged].index.values, [bin_ind] * len(df[flagged]),
                     yerr=None, xerr=0.5, elinewidth=20, ecolor='k', fmt='|', c='k',
                     markersize='20')

        # Draw circles around the flagged values in the full-resolution scatter plot.
        ax2.scatter(df[flagged].index.values,
                    df[flagged].loc[:, (df.columns.levels[0][1+bin_ind], 'k_pqei')].values,
                    marker='o',
                    # s=2*(1+bin_ind)**2, # With this, we get circles growing linearly in radius
                    s=20,
                    edgecolors=pu.CLRS['ref_1'],  # We color each circle manually. No cmap !!!
                    linewidth=0.5, facecolor='none', zorder=0)

    # Add the 0 line for reference.
    ax2.axhline(0, c='k', ls='-', lw=1)
    ax3.axhline(0, c='k', ls='-', lw=1)

    # Set the proper axis labels, etc ...
    for this_ax in [ax1]:
        this_ax.set_xlim((-0.5, len(df)-0.5))

    # Deal with the units, if warranted
    if unit is None:
        unit = ''
    else:
        unit = ' ['+pu.fix_txt(unit)+']'

    ax1.set_ylabel(r'$m$')
    ax2.set_ylabel(r'$k^{p,q}_{e,i}$')
    ax3.set_ylabel(r'$\Delta^{p,q}_{e,i}$' + unit)
    ax4.set_ylabel(r'$\sigma(\Delta^{p,q}_{e,i})$' + unit)
    ax4.set_xlabel(r'$i$')

    ax1.set_ylim((-0.5 + n_bins, -0.5))
    ax1.set_yticks(np.arange(0, n_bins, 1))
    ax1.set_yticklabels([r'%i' % (m_val) for m_val in df.columns.levels[0][1:]])
    # Hide x tick labels where I don't need them. Mind the fancy way to do it because of sharex.
    for this_ax in [ax1, ax2, ax3]:
        plt.setp(this_ax.get_xticklabels(), visible=False)
    ax1.tick_params(which='minor', axis='y', left=False, right=False)

    ax2.set_ylim((-6, 6))
    # Add the plot labels, if warranted.
    if left_label is not None:
        ax1.text(0, 1.1, pu.fix_txt(left_label), fontsize='small',
                 verticalalignment='bottom', horizontalalignment='left',
                 transform=ax1.transAxes)
    if right_label is not None:
        pu.add_var_and_k(ax1, var_name=right_label, offset=1.1)

    # Add the source for the plot
    pu.add_source(fig)

    # Save the figure
    pu.fancy_savefig(fig, fn_core='k-pqei', **kwargs)

    return fig
