"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level plotting for the UAII2022 recipe
"""

# Import general Python packages
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import transforms

# Import dvas modules and classes
from dvas.logger import log_func_call
from dvas.data.data import MultiProfile, MultiGDPProfile, MultiCWSProfile, MultiDeltaProfile
from dvas.hardcoded import PRF_IDX, PRF_TDT, PRF_ALT, PRF_VAL, PRF_UCU, PRF_UCS, PRF_UCT
from dvas.hardcoded import TAG_DTA, TAG_GDP, TAG_CWS
from dvas.hardcoded import MTDTA_PBLH, MTDTA_TROPOPAUSE, MTDTA_UTLSMIN, MTDTA_UTLSMAX
from dvas.data.data import MultiRSProfile
from dvas.tools.gdps import correlations as dtgc
from dvas.plots import utils as dpu
from dvas.plots import gdps as dpg
from dvas.dvas import Database as DB

# Import from dvas_recipes
from ..errors import DvasRecipesError
from .. import dynamic
from ..recipe import for_each_flight, for_each_var
from .. import utils as dru
from . import tools

# Setup local logger
logger = logging.getLogger(__name__)


@for_each_flight
@log_func_call(logger, time_it=True)
def flight_overview(start_with_tags, label='mid', show=None):
    """ Create an "overview" plot of all the recipe variables for a given flight.

    Args:
        start_with_tags (str|list of str): tag names for the search query into the
            database.
        label (str, optional): label of the plot legend. Defaults to 'mid'.
        show (bool, optional): if set, overrides the default dvas rule about whether to show the
            plot, or not. Defaults to None.

    """

    # Format the tags
    tags = dru.format_tags(start_with_tags)

    # Extract the flight info
    (fid, eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    filt = tools.get_query_filter(tags_in=tags + [eid, rid], tags_out=None)

    # The plot will have different number of rows depending on the number of variables.
    # Let's define some hardcoded heights, such that the look is always consistent
    top_gap = 0.4  # inch
    bottom_gap = 0.7  # inch
    plot_height = 1.3  # inch
    plot_gap = 0.05  # inch
    fig_height = (top_gap + bottom_gap + plot_height * len(dru.cws_vars())) /\
        (1 - plot_gap * (len(dynamic.ALL_VARS)-1))

    fig = plt.figure(figsize=(dpu.WIDTH_TWOCOL, fig_height))

    # Use gridspec for a fine control of the figure area.
    fig_gs = gridspec.GridSpec(len(dru.cws_vars()), 1,
                               height_ratios=[1]*len(dru.cws_vars()), width_ratios=[1],
                               left=0.085, right=0.87,
                               bottom=bottom_gap/fig_height, top=1-top_gap/fig_height,
                               wspace=0.05, hspace=plot_gap)

    for (var_ind, var_name) in enumerate(dru.cws_vars()):

        # Make x a shared axis if warranted
        if var_ind > 0:
            this_ax = fig.add_subplot(fig_gs[var_ind, 0], sharex=fig.axes[0])
        else:
            this_ax = fig.add_subplot(fig_gs[var_ind, 0])

        # Reset the color cycle for each plot, so that a given RS model always has the same color
        # Adapted from the reply of pelson and gg349 on StackOverflow:
        # https://stackoverflow.com/questions/24193174
        plt.gca().set_prop_cycle(None)

        # Extract the data from the db
        rs_prfs = MultiRSProfile()
        rs_prfs.load_from_db(filt, var_name, dynamic.INDEXES[PRF_TDT],
                             alt_abbr=dynamic.INDEXES[PRF_ALT])
        rs_prfs.sort()  # Sorting is important to make sure I have the same colors for each plot

        # Setup some dummy limits, to keep track of the actual ones as I go.
        xmin = np.infty
        xmax = -np.infty

        # Start looping over all the profiles, and plot them one-by-one.
        for (prf_ind, prf) in enumerate(rs_prfs):

            # Make this plot as a function of the index i, so that synchronized profiles will
            # show up cleanly.
            x = getattr(prf, PRF_VAL).index.get_level_values(PRF_IDX)
            y = getattr(prf, PRF_VAL).values

            if var_name == 'wdir':
                x, y = dpu.wrap_wdir_curve(x, y)

            this_ax.plot(x, y, '-', lw=0.7, label='|'.join(rs_prfs.get_info(label)[prf_ind]))

            xmin = np.min([xmin, np.min(x)])
            xmax = np.max([xmax, np.max(x)])

        # Set tight x limits
        this_ax.set_xlim((xmin, xmax))

        # Once only: show the legend and the edt/eid/rid info
        if var_ind == 0:
            dpu.add_edt_eid_rid(this_ax, rs_prfs)
            dpu.fancy_legend(this_ax, label=label)
            this_ax.text(1, 1.03, 'tags: ' + ' / '.join(tags), fontsize='small',
                         verticalalignment='bottom', horizontalalignment='right',
                         transform=this_ax.transAxes)

        # Hide the tick labels except for the bottom-most plot
        if var_ind < len(dru.cws_vars())-1:
            plt.setp(this_ax.get_xticklabels(), visible=False)

        # Set the ylabel:
        ylbl = rs_prfs.var_info[PRF_VAL]['prm_plot']
        ylbl += f' [{rs_prfs.var_info[PRF_VAL]["prm_unit"]}]'
        # Include the ylabel as text, to have it left-aligned with all other subplots
        this_ax.text(-0.1, 0.5, dpu.fix_txt(ylbl), ha='left', va='center',
                     transform=this_ax.transAxes, rotation=90)
        if var_name == 'wdir':
            this_ax.set_ylim((0, 360))
            this_ax.set_yticks([0, 180])

    # Set the label for the last plot only
    this_ax.set_xlabel(r'$i$')

    # Add the source
    dpu.add_source(fig)

    # Save it all
    dpu.fancy_savefig(fig, 'flight_overview', fn_prefix=dynamic.CURRENT_STEP_ID,
                      fn_suffix=dru.fn_suffix(fid=fid, eid=eid, rid=rid, tags=tags),
                      fmts=dpu.PLOT_FMTS, show=show)


def covmat_stats(covmats):
    """ Takes a closer look at the *true* covariance matrix computed by dvas for a combined profile.

    Looks in particular at the error one does by ignoring it and assuming the combined profile
    uncertainties behave like a ucu, ucs, or uct types or uncertainties.

    Args:
        covmats (dict): the outcome of dvas.tools.gdps.combine()[1], i.e. the dict of covariance
            matrices.
    """

    # Setup a dict to store the "theoretical" covariance matrices
    th_covmats = {}
    # And also the error arrays
    errors = {}
    # How many covariance elements does the real matrix have (size-diag), i.e. how many were
    # computed ? This is directly dependant on the chunk size
    perc_covelmts_comp = {}

    # Prepare the bins as well
    bins = [-100, -20, -15] + list(np.linspace(-10, -1, 10)) + [-0.1, 0.1] +\
        list(np.linspace(1, 10, 10)) + [15, 20, 100]

    # Loop through all the uncertainty types
    for uc_name in [PRF_UCS, PRF_UCT, PRF_UCU]:

        # How many points are associated to the matrix ?
        npts = covmats[uc_name].shape[0]

        # Build matrices of indexes
        inds = np.arange(0, npts, 1)
        # Which position are off-diagonal ?
        off_diag = ~np.eye(npts, npts, dtype=bool)
        # Out of these, which ones are "valid", i.e. not NaN ?
        valids = off_diag * ~np.isnan(covmats[uc_name])

        # For the uncorrelated uncertainties, all the off-diagonal elements should always be 0
        # (unless they are NaNs). Let's issue a log-ERROR message if this is not the case.
        if uc_name in [PRF_UCU]:
            if np.any(covmats[uc_name][valids] != 0):
                logger.error("Non-0 off-diagonal elements of covarience matrix [%s].",
                             uc_name)

        # Now, let's compute the theoretical covariance matrix.
        # This is the covariance of the different elements of the combined profile with itself.
        # As such, it doesn't matter what the mid, rid, eid, oid actaully are - their just the same
        # for all the points in the profile.
        cc_mat = dtgc.corr_coeff_matrix(uc_name, inds, oids=np.ones(npts), mids=np.ones(npts),
                                        rids=np.ones(npts), eids=np.ones(npts))

        # And now get the uncertainties from the diagonal of the covariance matrix ...
        sigmas = np.atleast_2d(np.sqrt(covmats[uc_name].diagonal()))
        # ... turn them into a masked array ...
        sigmas = np.ma.masked_invalid(sigmas)
        # ... and combine them with the correlation coefficients. Mind the mix of Hadamard and dot
        # products to get the correct mix !
        th_covmats[uc_name] = np.multiply(cc_mat, np.ma.dot(sigmas.T, sigmas))

        # Having done so, we can compute the relative error (in %) that one does by reconstructing
        # the covariance matrix assuming it is exactly of the given ucs/t/u type.
        errors[uc_name] = th_covmats[uc_name][valids] / covmats[uc_name][valids] - 1
        errors[uc_name] *= 100

        # If more than 10% of the covariance points have an error greater than 10%, raise a
        # log-error.
        if (tmp := len(errors[uc_name][np.abs(errors[uc_name]) > 10])) > 0.1 * len(valids[valids]):

            msg = 'of the verifiable theoretical covariance matrix elements differ by more than'
            msg = msg + r'10\% of the true value.'
            logger.error(r'%s \% %s', (tmp, msg))

        errors[uc_name] = np.histogram(errors[uc_name],
                                       bins=bins,
                                       density=False)[0] / len(valids[valids]) * 100

        # Finally, store the percentage of covariance elements that can be checked,
        # i.e that were computed.
        perc_covelmts_comp[uc_name] = len(valids[valids]) / npts**2 * 100

    # Now, let's make a histogram plot of this information
    fig = plt.figure(figsize=(dpu.WIDTH_ONECOL, 5))

    # Use gridspec for a fine control of the figure area.
    fig_gs = gridspec.GridSpec(1, 1,
                               height_ratios=[1], width_ratios=[1],
                               left=0.15, right=0.95,
                               bottom=0.2, top=0.9,
                               wspace=0.05, hspace=0.05)

    ax0 = plt.subplot(fig_gs[0, 0])

    for uc_name in ['ucs', 'uct', 'ucu']:
        ax0.plot(np.diff(bins), errors[uc_name], '-', drawstyle='steps-mid',
                 label=rf'{uc_name} ({perc_covelmts_comp[uc_name]:.2f}\%)')

    plt.legend()
    ax0.set_xlim((-25, 25))

    ax0.set_xlabel(r'$(V_{i\neq j}^{\rm th}/{V_{i\neq j}}-1)\times 100$')
    ax0.set_ylabel(r'Normalized number count [\%]', labelpad=10)

    ax0.set_ylim((0, 100))

    # Add the k-level
    dpu.add_var_and_k(ax0, var_name=dynamic.CURRENT_VAR)

    # Add the source
    dpu.add_source(fig)

    # Save it all
    # Get the event id and rig id
    (fid, eid, rid) = dynamic.CURRENT_FLIGHT
    dpu.fancy_savefig(fig, f'covmat_check_chunk-size-{dynamic.CHUNK_SIZE}',
                      fn_prefix=dynamic.CURRENT_STEP_ID,
                      fn_suffix=dru.fn_suffix(fid=fid, eid=eid, rid=rid, var=dynamic.CURRENT_VAR),
                      fmts=dpu.PLOT_FMTS, show=None)


@for_each_var()
@for_each_flight
@log_func_call(logger, time_it=True)
def inspect_cws(gdp_start_with_tags, cws_start_with_tags):
    """ Create a series of CWS-related plot for inspection purposes.

    Args:
        gdp_start_with_tags (str|list of str): gdp tag names for the search query into the database.
        cws_start_with_tags (str|list of str): cws tag names for the search query into the database.
    """

    # Format the tags
    gdp_tags = dru.format_tags(gdp_start_with_tags)
    cws_tags = dru.format_tags(cws_start_with_tags)

    # This recipe step should be straight forward. I first need to extract the GDPs, then the CWS,
    # and simply call the dedicated plotting routine.

    # Get the event id and rig id
    (fid, eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    gdp_filt = tools.get_query_filter(tags_in=gdp_tags + [eid, rid, TAG_GDP],
                                      tags_out=None)
    cws_filt = tools.get_query_filter(tags_in=cws_tags + [eid, rid, TAG_CWS],
                                      tags_out=None)

    # Load the GDP profiles
    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(gdp_filt, dynamic.CURRENT_VAR,
                          tdt_abbr=dynamic.INDEXES[PRF_TDT],
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                          uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                          ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                          inplace=True)
    gdp_prfs.sort(inplace=True)

    # Idem for the CWS
    cws_prfs = MultiCWSProfile()
    cws_prfs.load_from_db(cws_filt, dynamic.CURRENT_VAR,
                          tdt_abbr=dynamic.INDEXES[PRF_TDT],
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                          uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                          ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                          inplace=True)
    cws_prfs.sort(inplace=True)

    # We can now create a GDP vs CWS plot ...
    dpg.gdps_vs_cws(gdp_prfs, cws_prfs, show=None,
                    fn_prefix=dynamic.CURRENT_STEP_ID,
                    fn_suffix=dru.fn_suffix(fid=fid, eid=eid, rid=rid, tags=gdp_tags,
                                            var=dynamic.CURRENT_VAR))
    # And a uc_budget plot
    dpg.uc_budget(gdp_prfs, cws_prfs, show=None,
                  fn_prefix=dynamic.CURRENT_STEP_ID,
                  fn_suffix=dru.fn_suffix(fid=fid, eid=eid, rid=rid, tags=gdp_tags,
                                          var=dynamic.CURRENT_VAR))


@for_each_var(incl_wvec=True)
@for_each_flight
@log_func_call(logger, time_it=False)
def participant_preview(prf_tags, cws_tags, dta_tags, mids=None):
    """ Create the official per-flight preview diagram for the participants.

    Args:
        prf_tags (str, list): which tags to use to identify Profiles in the DB.
        cws_tags (str, list): which tags to use to identify CWS in the DB.
        dta_tags (str, list): which tags to use to identify DeltaProfiles in the DB.
        mids (list, optional): list of 'mid' to process. Defaults to None = all
    """

    # Get the event id and rig id
    (fid, eid, rid) = dynamic.CURRENT_FLIGHT

    # Format the tags
    prf_tags = dru.format_tags(prf_tags)
    cws_tags = dru.format_tags(cws_tags)
    dta_tags = dru.format_tags(dta_tags)

    # Very well, let us first extract the list of 'mid', if they have not been provided
    db_view = DB.extract_global_view()
    if mids is None:
        mids = db_view.mid.unique().tolist()

    # Basic sanity check of mid
    if not isinstance(mids, list):
        raise DvasRecipesError(f'I need a list of mids, not: {mids}')

    # Prepare the search queries
    prf_filt = tools.get_query_filter(tags_in=prf_tags + [eid, rid],
                                      tags_out=[TAG_CWS],
                                      mids=mids)

    cws_filt = tools.get_query_filter(tags_in=cws_tags + [eid, rid, TAG_CWS],
                                      tags_out=None)

    dta_filt = tools.get_query_filter(tags_in=dta_tags + [eid, rid, TAG_DTA],
                                      tags_out=None, mids=mids)

    # Query these different Profiles and CWS. These exists only for the CWS variables
    if dynamic.CURRENT_VAR in dru.cws_vars():
        prfs = MultiProfile()
        prfs.load_from_db(prf_filt, dynamic.CURRENT_VAR, dynamic.INDEXES[PRF_ALT],
                          inplace=True)

        logger.info('Loaded %i Profiles for mids: %s', len(prfs), prfs.get_info('mid'))

        cws_prfs = MultiCWSProfile()
        cws_prfs.load_from_db(cws_filt, dynamic.CURRENT_VAR, dynamic.INDEXES[PRF_TDT],
                              alt_abbr=dynamic.INDEXES[PRF_ALT],
                              ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                              uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                              ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                              inplace=True)
        logger.info('Loaded %i CWS.', len(prfs))

    dta_prfs = MultiDeltaProfile()
    dta_prfs.load_from_db(dta_filt, dynamic.CURRENT_VAR,
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                          uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                          ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                          inplace=True)

    logger.info('Loaded %i DeltaProfiles for mids: %s', len(dta_prfs), dta_prfs.get_info('mid'))

    if dynamic.CURRENT_VAR in dru.cws_vars():
        # Make sure I have the same number of prfs and dta_prfs. Else, it most likely means that
        # some dta were not computed.
        if len(dta_prfs) < len(prfs):
            raise DvasRecipesError('Found less dta_prfs than prfs. ' +
                                   'Have all the DeltaProfiles been computed ?')
        if len(dta_prfs) > len(prfs):
            raise DvasRecipesError('Found more dta_prfs than prfs.')

    # Very well, let's now loop through these, and generate the plot
    for (p_ind, dta) in enumerate(dta_prfs):

        # What is the mid of this prf ?
        mid = dta.info.mid
        logger.info('Processing mid: %s', mid)

        # Extract the corresponding profile
        if dynamic.CURRENT_VAR in dru.cws_vars():
            prf = prfs[p_ind]
            # dta_prfs should have the same order as prfs. But let's make sure.
            if prf.info.mid != mid:
                raise DvasRecipesError(
                    f'"mid" mismatch between prfs and dta_prfs: {prf.info.mid} vs {mid}')

            # Let's make some more sanity checks, to avoid bad surprises ...
            if len(cws_prfs) != 1:
                raise DvasRecipesError(f'I got {len(cws_prfs)} != 1 CWS profiles ?!')

        # Let's extract the corresponding SRN and PID
        srn = db_view.srn.values[db_view.oid.values == dta_prfs[p_ind].info.oid][0]
        pid = db_view.pid.values[db_view.oid.values == dta_prfs[p_ind].info.oid][0]

        # Start the plotting
        fig = plt.figure(figsize=(dpu.WIDTH_TWOCOL, 5.5))

        # Create a gridspec structure
        gs_info = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], width_ratios=[1],
                                    left=0.09, right=0.87, bottom=0.12, top=0.93,
                                    wspace=0.5, hspace=0.1)

        # Create the axes - one for the profiles, and one for uctot, ucs, uct, ucu
        ax0 = fig.add_subplot(gs_info[0, 0])
        ax1 = fig.add_subplot(gs_info[1, 0], sharex=ax0)

        # Extract the DataFrames
        if dynamic.CURRENT_VAR in dru.cws_vars():
            prf_pdf = prfs.get_prms([PRF_ALT, PRF_VAL])[p_ind]
            cws_pdf = cws_prfs.get_prms([PRF_ALT, PRF_VAL, 'uc_tot'])[0]
        dta_pdf = dta_prfs.get_prms([PRF_ALT, PRF_VAL, 'uc_tot'])[p_ind]

        if dynamic.CURRENT_VAR in dru.cws_vars():
            # Show the delta = 0 line
            ax1.plot(cws_pdf.loc[:, PRF_ALT].values, cws_pdf.loc[:, PRF_VAL].values*0,
                     lw=0.4, ls='-', c='darkorchid')

            # Very well, let us plot all these things.
            # First, plot the profiles themselves
            ax0.plot(cws_pdf.loc[:, PRF_ALT].values, prf_pdf.loc[:, PRF_VAL].values,
                     lw=0.4, ls='-', drawstyle='steps-mid', c='k', alpha=1,
                     label=mid[0])
            ax0.plot(cws_pdf.loc[:, PRF_ALT].values, cws_pdf.loc[:, PRF_VAL].values,
                     lw=0.4, ls='-', drawstyle='steps-mid', c='darkorchid', alpha=1, label='CWS')
            ax0.fill_between(cws_pdf.loc[:, PRF_ALT].values,
                             prfs.get_prms(PRF_VAL).max(axis=1).values,
                             prfs.get_prms(PRF_VAL).min(axis=1).values,
                             facecolor=(0.8, 0.8, 0.8), step='mid', edgecolor='none',
                             label='All sondes')

        # Next plot the uncertainties
        ax1.plot(dta_pdf.loc[:, PRF_ALT], dta_pdf.loc[:, PRF_VAL],
                 alpha=1, drawstyle='steps-mid', color='k', lw=0.4, label=mid[0])
        ax1.fill_between(dta_pdf.loc[:, PRF_ALT].values,
                         dta_prfs.get_prms(PRF_VAL).max(axis=1).values,
                         dta_prfs.get_prms(PRF_VAL).min(axis=1).values,
                         facecolor=(0.8, 0.8, 0.8), step='mid', edgecolor='none',
                         label='All sondes')

        # Display the location of the tropopause and the PBL
        for (loi, symb) in [(MTDTA_TROPOPAUSE, r'$\prec$'), (MTDTA_PBLH, r'$\simeq$'),
                            (MTDTA_UTLSMIN, r'$\top$'), (MTDTA_UTLSMAX, r'$\bot$')]:
            if loi not in dta.info.metadata.keys():
                logger.warning('"%s" not found in metadata.', loi)
                continue

            loi_gph = float(dta.info.metadata[loi].split(' ')[0])

            for ax in [ax0, ax1]:
                ax.axvline(loi_gph, ls=':', lw=1, c='k')
            trans = transforms.blended_transform_factory(ax0.transData, ax0.transAxes)
            ax0.text(loi_gph, 0.95, symb, transform=trans, ha='center', va='top',
                     rotation=90, bbox=dict(boxstyle='square', fc="w", ec="none", pad=0.1))

        # Set the axis labels
        ylbl0 = r'$x_{e,i}$ ' + f'[{dta_prfs.var_info[PRF_VAL]["prm_unit"]}]'
        ylbl1 = r'$\delta_{e,i}$'
        ylbl1 += f' [{dta_prfs.var_info[PRF_VAL]["prm_unit"]}]'
        altlbl = dta_prfs.var_info[PRF_ALT]['prm_plot']
        altlbl += f' [{dta_prfs.var_info[PRF_ALT]["prm_unit"]}]'

        # Plot ylabels as text, to have them left-aligned accross sub-plots
        ax0.text(-0.1, 0.5, dpu.fix_txt(ylbl0), ha='left', va='center',
                 transform=ax0.transAxes, rotation=90)
        ax1.text(-0.1, 0.5, dpu.fix_txt(ylbl1), ha='left', va='center',
                 transform=ax1.transAxes, rotation=90)
        ax1.set_xlabel(dpu.fix_txt(altlbl))

        # For the delta curve, set the scale for this specific mid (in case the rest of the sondes
        # behave very badly). This seems convoluted, but accounts for cases when ymin/ymax
        # are negative. It reproduces the default behavior of autoscale.
        ymin = dta_pdf.loc[:, PRF_VAL].min()
        ymax = dta_pdf.loc[:, PRF_VAL].max()
        if ~np.isnan(ymin) and ~np.isnan(ymax):  # If the delta is full or NaN, this may happen ...
            ax1.set_ylim((ymin-0.05*np.abs(ymax-ymin),
                          ymax+0.05*np.abs(ymax-ymin)))
        for ax in [ax0]:
            plt.setp(ax.get_xticklabels(), visible=False)

        # Add the legend
        if dynamic.CURRENT_VAR in dru.cws_vars():
            dpu.fancy_legend(ax0, '')
        else:
            dpu.fancy_legend(ax1, '')
        # Add the edt/eid/rid info
        dpu.add_edt_eid_rid(ax0, dta_prfs)

        # Add the source for the plot
        dpu.add_source(fig)

        dpu.add_var_and_k(ax0, mid='+'.join(mid)+rf' \#{pid} ({srn})',
                          var_name=dta_prfs.var_info[PRF_VAL]['prm_plot'], k=None)

        # Save it
        fn_suf = dru.fn_suffix(fid=fid, eid=eid, rid=rid, tags=None, mids=mid, pids=[pid],
                               var=dynamic.CURRENT_VAR)
        dpu.fancy_savefig(fig, fn_core='pp', fn_suffix=fn_suf, fn_prefix=dynamic.CURRENT_STEP_ID)
