"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Plotting functions related to the gruan submodule.

"""

# Import from Python packages
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import from this package
from ..errors import  DvasError
from ..logger import log_func_call
from ..logger import plots_logger as logger
from ..hardcoded import PRF_REF_INDEX_NAME, PRF_REF_VAL_NAME, PRF_REF_ALT_NAME, PRF_REF_TDT_NAME
from ..hardcoded import PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME
from . import utils as pu

@log_func_call(logger)
def gdps_vs_cws(gdp_prfs, cws_prf, k_lvl=1, index_name='tdt', label='mid', **kwargs):
    """ Makes a plot comparing different GDPs with their associated combined working measurement
    standard.

    All profiles must imperatively be fully synchronized.

    Args:
        gdp_prfs (dvas.data.data.MultiGDPProfile): the GDPs
        cws_prf (dvas.data.data.MultiGDPProfile): the combined working standards, for example
            generated by dvas.tools.gruan.combine_gdps().
        k_lvl (int|float, optional): k-level for the uncertainty. Defaults to 1.
        index_name (str, optional): reference variables for the plots, either 'tdt' or 'alt'.
            Defaults to 'tdt'.
        label (str, optional): label of the plot legend. Defaults to 'mid'.
        **kwargs: these get fed to the dvas.plots.utils.fancy_savefig() routine.

    Returns:
        matplotlib.pyplot.figure: the figure instance
    """

    # Start the plotting
    fig = plt.figure(figsize=(pu.WIDTH_TWOCOL, 5.0))

    # Create a gridspec structure
    gs_info = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[1],
                                left=0.09, right=0.87, bottom=0.15, top=0.93,
                                wspace=0.5, hspace=0.1)

    # Create the axes - one for the profiles, and one for the Delta
    ax0 = fig.add_subplot(gs_info[0, 0])
    ax1 = fig.add_subplot(gs_info[1, 0], sharex=ax0)

    # Extract the DataFrames from the MultiGDPProfile instances
    cws = cws_prf.get_prms([PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME,
                            PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME,
                            'uc_tot'])[0]
    gdps = gdp_prfs.get_prms([PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME,
                            PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME,
                            'uc_tot'])

    # What are my reference values ? Here I need to differentiate two cases if the user asks for
    # 'tdt' or 'alt'. These two index are put into regular columns by the get_prms() method !
    if index_name in cws.index.names:
        x = cws.index.get_level_values(index_name).values
    else:
        x = cws[index_name]
        if index_name == PRF_REF_TDT_NAME:
            # I need to do a unit conversion for the time deltas ... having timedelta64[ns] is
            # not being supported by matplotlib.
            x = x.dt.total_seconds()
        else:
            x = x.values

    # Very well, let us plot all these things.
    for gdp_ind in range(len(gdps.columns.levels[0])):

        gdp = gdps[gdp_ind]

        # Let us make sure that all the profiles are synchronized by checking the profile lengths
        # This is not fool proof, but it is a start. I could also check for the sync tag, but it
        # would be less strict as a check.
        if len(cws) != len(gdps):
            raise DvasError('Ouch! GDPS and CWS are not synchronized. I cannot plot this.')

        # First, plot the profiles themselves
        ax0.plot(x, gdp[PRF_REF_VAL_NAME], lw=0.5, ls='-', drawstyle='steps-mid',
                 label='|'.join(gdp_prfs.get_info(label)[gdp_ind]))
        ax0.fill_between(x, gdp[PRF_REF_VAL_NAME]-k_lvl*gdp['uc_tot'],
                         gdp[PRF_REF_VAL_NAME]+k_lvl*gdp['uc_tot'], alpha=0.3, step='mid')

        # And below, plot the Deltas with respect to the CWS
        delta = gdp[PRF_REF_VAL_NAME].values-cws[PRF_REF_VAL_NAME].values
        ax1.plot(x, delta, drawstyle='steps-mid', lw=0.5, ls='-')
        ax1.fill_between(x, delta-k_lvl*gdp['uc_tot'], delta+k_lvl*gdp['uc_tot'], alpha=0.3,
                         step='mid')

    # Then also plot the CWS uncertainty
    ax0.plot(x, cws['val'], color=pu.CLRS['cws_1'], lw=0.5, ls='-', drawstyle='steps-mid',
             label='CWS')
    ax0.fill_between(x, cws[PRF_REF_VAL_NAME]-k_lvl*cws['uc_tot'],
                     cws[PRF_REF_VAL_NAME]+k_lvl*cws['uc_tot'], alpha=0.3, step='mid',
                     color=pu.CLRS['cws_1'])

    #ax1.fill_between(x, -k_lvl*cws['uc_tot'], +k_lvl*cws['uc_tot'], alpha=0.3, step='mid',
    #                 color=pu.CLRS['cws_1'])

    ax1.plot(x, -k_lvl*cws['uc_tot'], lw=0.5, drawstyle='steps-mid',
             color='k')
    ax1.plot(x, +k_lvl*cws['uc_tot'], lw=0.5, drawstyle='steps-mid',
             color='k')

    # Make it look pretty

    ylbl = cws_prf.var_info[PRF_REF_VAL_NAME]['prm_name']
    ylbl += ' [{}]'.format(cws_prf.var_info[PRF_REF_VAL_NAME]['prm_unit'])

    if index_name == PRF_REF_INDEX_NAME:
        xlbl = r'$i$'
    else:
        xlbl = cws_prf.var_info[index_name]['prm_name']
        xlbl += ' [{}]'.format(cws_prf.var_info[index_name]['prm_unit'])


    ax0.set_ylabel(pu.fix_txt(ylbl))
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax1.set_xlabel(pu.fix_txt(xlbl))
    ax1.set_ylabel(pu.fix_txt(r'$\Delta$' + ylbl))

    ax0.set_xlim((np.nanmin(x), np.nanmax(x)))

    # Add the legend
    pu.fancy_legend(ax0, label)

    # Add the edt/eid/rid info
    pu.add_edt_eid_rid(ax0, cws_prf)

    # Add the k-level if warranted
    ax0.text(1, 1.03, r'$k={}$'.format(k_lvl), fontsize='small',
             verticalalignment='bottom', horizontalalignment='right',
             transform=ax0.transAxes)

    # Add the source for the plot
    pu.add_source(fig)

    pu.fancy_savefig(fig, fn_core='gdps-vs-cws', **kwargs)

@log_func_call(logger)
def plot_gdps(fn_list, ref_var='time', tag=None, save_loc=None):
    """ Make a basic plot to compare all GDP variables at a glance.

    Args:
       fn_list (list of pathlib.Path): list of GDP filenames.
       ref_var (str): name of the variable to use for the abscissa of the plots.
       tag (str, optional): str to add to the filename for tagging.
       save_loc (pathlib.Path, optional): if set, overrides the dvas location to save the plot.

    Note:
        If `ref_var != 'time'`, the differences between the different profiles will not be computed.

    TODO:
        * Change the input to a series of TimeProfileManager instaed of a list of filenames.
        * Prettify the GPS plot using underlying country broders (ideally NOT via cartopy, given
          its heavy requirements).
    """

    raise Exception("Ouch ! This function needs to be brought back from the dead.")

    ## Extract the data from the nc files
    #full_data = [nc.Dataset(this_fn, 'r') for this_fn in fn_list]
    #
    ## One issue with raw nc files is that different profiles may have different lengths.
    ## This is a real issue for the plotting, so let's find the minimum common length.
    ## This should be avoided through the use of TimeProfileManager, that should all have a common
    ## length for profiles of a given multi-flight.
    #n_steps_min = np.min([len(data.variables['time'][:]) for data in full_data])
    #
    ## Some sanity checks first
    #if tag is None:
    #    tag = ''
    #elif tag[0] != '_': # To keep the filename good looking.
    #    tag = '_' + tag
    #
    #if save_loc is None:
    #    if (save_loc := path_var.plot_output_path) is None:
    #        raise Exception()
    #elif isinstance(save_loc, str): # The user fed a string ... be nice and assume this is a path.
    #    save_loc = Path(save_loc)
    #elif not isinstance(save_loc, PurePath):
    #    raise Exception('Ouch ! I was expecting a pathlib.Path type for save_loc, not: %s' %
    #                    (type(save_loc)))
    #
    ## Start the plotting
    #plt.close(51)
    #plt.figure(51, figsize=(pu.WIDTH_TWOCOL, 9.5))
    #
    ## Create a gridspec structure for what will be a busy multi-panel plot.
    #gs_info = gridspec.GridSpec(9, 4,
    #                            height_ratios=[1, 0.5, 0.12, 1, 0.5, 0.12, 1, 0.5, 0.12],
    #                            width_ratios=[1, 1, 1, 1],
    #                            left=0.08, right=0.98, bottom=0.05, top=0.93,
    #                            wspace=0.5, hspace=0.1)
    #
    ## First, create the axes for the main variables: temp, rh, press, wspeed
    ## Good luck to anyone trying to understand (let alone adjust!) this terrifingly-awesome
    ## gridpec structure ...
    #var_axs = [plt.subplot(gs_info[i%6 + (i%6)//2 + (i//6)*3,
    #                               2*(i//6):2*(i//6)+2]) for i in range(10)]
    ## Then,  create the axes for the somewhat different wdir and lat-lon plots
    #wdir_ax = plt.subplot(gs_info[0:2, 2], projection='polar')
    #wdir_ax.set_theta_offset(np.pi/2) # Puts North up
    #wdir_ax.set_theta_direction(-1) #Puts 270deg = West to the left
    #gps_ax = plt.subplot(gs_info[0:2, 3])
    #
    ## Extract the primary variables I care about
    #for (n_ind, name) in enumerate(['temp', 'rh', 'press', 'wspeed', 'alt']):
    #    # Now loop through all the data files
    #    for (_, data) in enumerate(full_data):
    #
    #        # Plot the error zone
    #        with np.errstate(invalid='ignore'):
    #            # The previous line is needed to hide RunTimeWarnings caused by NaNs wreaking havoc
    #            # with fill_between.
    #            var_axs[2*n_ind].fill_between(data.variables[ref_var][:n_steps_min],
    #                                          data.variables[name][:n_steps_min] -
    #                                          data.variables[name + '_uc'][:n_steps_min],
    #                                          data.variables[name][:n_steps_min] +
    #                                          data.variables[name + '_uc'][:n_steps_min],
    #                                          step='mid', alpha=0.2)
    #        # Plot the measured profile
    #        var_axs[2*n_ind].plot(data.variables[ref_var][:n_steps_min],
    #                              data.variables[name][:n_steps_min],
    #                              marker='None', linestyle='-', drawstyle='steps-mid',
    #                              lw=0.7, markersize=1)
    #
    #        # And also plot the differences with the first profile of the group
    #        # This makes sense only if ref_var == 'time'. Else, some interpolation is needed ...
    #        # but this may not always work (e.g. if balloon goes up and down and up again).
    #        if ref_var == 'time':
    #            var_axs[2*n_ind+1].plot(data.variables[ref_var][:n_steps_min],
    #                                    data.variables[name][:n_steps_min] -
    #                                    full_data[0].variables[name][:n_steps_min],
    #                                    marker='None', linestyle='-', drawstyle='steps-mid',
    #                                    lw=0.7)
    #
    #        # Add the labels, and set some limits
    #        var_axs[2*n_ind].set_ylabel(name+'[%s]' % (pu.UNIT_LABELS[data.variables[name].units]))
    #        if ref_var == 'time':
    #            var_axs[2*n_ind+1].set_ylabel(r'$\Delta$ [%s]' %
    #                                          (pu.UNIT_LABELS[data.variables[name].units]))
    #        var_axs[2*n_ind].set_xlim((0, np.max(data.variables[ref_var][:n_steps_min])))
    #        var_axs[2*n_ind+1].set_xlim((0, np.max(data.variables[ref_var][:n_steps_min])))
    #
    #        # Hide the x-tick labels when needed
    #        var_axs[2*n_ind].set_xticklabels([])
    #        if name not in ['press', 'alt']:
    #            var_axs[2*n_ind+1].set_xticklabels([])
    #        else:
    #            if ref_var == 'time':
    #                var_axs[2*n_ind+1].set_xlabel(r'Time [s]')
    #            else:
    #                var_axs[2*n_ind+1].set_xlabel(ref_var+' [%s]' %
    #                                              (pu.UNIT_LABELS[data.variables[ref_var].units]))
    #
    ## Now deal with the non-standard plots
    #for (_, data) in enumerate(full_data):
    #    # Make a polar plot for the wind direction
    #    wdir_ax.plot(np.radians(data.variables['wdir'][:n_steps_min]),
    #                 data.variables[ref_var][:n_steps_min])
    #    wdir_ax.set_yticklabels([])
    #
    #    # Plot the GPS tracks
    #    # At some point, it could be nice to make this prettier ... with an underlying map ?
    #    gps_ax.plot(data.variables['lon'][:n_steps_min], data.variables['lat'][:n_steps_min])
    #    # Also show the starting point
    #    gps_ax.plot(data.variables['lon'][:][0], data.variables['lat'][:][0], marker='x', c='k')
    #
    ## Fine tune some of the look
    #wdir_ax.set_title(r'wdir', pad=20)
    #gps_ax.set_title(r'Lon. - Lat. [$^{\circ}$]')
    #
    #plt.savefig(save_loc / ('GDPs_%s%s.png' % (ref_var, tag)))
    #

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
    top_gap = 0.4 # inch
    bottom_gap = 0.7 # inch
    plot_1_height = 0.35*n_bins # inch
    plot_2_height = 2. # inch
    mid_gap = 0.05 # inch
    fig_height = top_gap + bottom_gap + plot_1_height + 3*plot_2_height + 3*mid_gap

    # Set up the scene ...
    fig = plt.figure(figsize=(pu.WIDTH_TWOCOL, fig_height))

    gs_info = gridspec.GridSpec(4, 1, height_ratios=[plot_1_height/plot_2_height, 1, 1, 1],
                                width_ratios=[1], left=0.08, right=0.98,
                                bottom=bottom_gap/fig_height, top=1-top_gap/fig_height,
                                wspace=0.02, hspace=mid_gap)

    ax1 = fig.add_subplot(gs_info[0, 0]) # A 2D plot of the incompatible points as a function of m
    ax2 = fig.add_subplot(gs_info[1, 0], sharex=ax1) # A scatter plot of k_pq^ei
    ax3 = fig.add_subplot(gs_info[2, 0], sharex=ax1) # A scatter plot of Delta_pq^ei
    ax4 = fig.add_subplot(gs_info[3, 0], sharex=ax1) # A scatter plot of sigma_pq^ei

    # First, let's plot the full-resolution data.
    ax2.scatter(df.index.values, df.loc[:, (0, 'k_pqei')], marker='o',
                facecolor=pu.CLRS['nan_1'], s=1, edgecolor=None, zorder=10)
    ax3.scatter(df.index.values, df.loc[:, (0, 'Delta_pqei')], marker='o',
                facecolor=pu.CLRS['nan_1'], s=1, edgecolor=None, zorder=10)
    ax4.scatter(df.index.values, df.loc[:, (0, 'sigma_pqei')], marker='o',
                facecolor=pu.CLRS['nan_1'], s=1, edgecolor=None, zorder=10)

    # For k, show the k=1, 2, 3 zones
    for k in [1,2,3]:
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
                    #s=2*(1+bin_ind)**2, # With this, we get circles growing linearly in radius
                    s=20,
                    edgecolors=pu.CLRS['ref_1'], # We color each circle manually. No cmap !!!
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
        ax1.text(0, 1.03, pu.fix_txt(left_label), fontsize='small',
                 verticalalignment='bottom', horizontalalignment='left',
                 transform=ax1.transAxes)
    if right_label is not None:
        ax1.text(1, 1.03, pu.fix_txt(right_label), fontsize='small',
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=ax1.transAxes)

    # Add the source for the plot
    pu.add_source(fig)

    # Save the figure
    pu.fancy_savefig(fig, fn_core='k-pqei', **kwargs)

    return fig
