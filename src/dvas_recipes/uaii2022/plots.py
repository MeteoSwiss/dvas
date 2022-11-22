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
from dvas.hardcoded import PRF_IDX, PRF_TDT, PRF_ALT, PRF_VAL, PRF_UCU, PRF_UCR, PRF_UCS, PRF_UCT
from dvas.hardcoded import TAG_DTA, TAG_GDP, TAG_CWS, MTDTA_PBL, MTDTA_TROPOPAUSE
from dvas.data.data import MultiRSProfile
from dvas.tools.gdps import utils as dtgu
from dvas.plots import utils as dpu
from dvas.plots import gdps as dpg
from dvas.plots import dtas as dpd
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
    filt = tools.get_query_filter(tags_in=tags+[eid, rid], tags_out=dru.rsid_tags(pop=tags))

    # The plot will have different number of rows depending on the number of variables.
    # Let's define some hardcoded heights, such that the look is always consistent
    top_gap = 0.4  # inch
    bottom_gap = 0.7  # inch
    plot_height = 1.3  # inch
    plot_gap = 0.05  # inch
    fig_height = (top_gap + bottom_gap + plot_height * len(dynamic.ALL_VARS)) /\
        (1 - plot_gap * (len(dynamic.ALL_VARS)-1))

    fig = plt.figure(figsize=(dpu.WIDTH_TWOCOL, fig_height))

    # Use gridspec for a fine control of the figure area.
    fig_gs = gridspec.GridSpec(len(dynamic.ALL_VARS), 1,
                               height_ratios=[1]*len(dynamic.ALL_VARS), width_ratios=[1],
                               left=0.085, right=0.87,
                               bottom=bottom_gap/fig_height, top=1-top_gap/fig_height,
                               wspace=0.05, hspace=plot_gap)

    for (var_ind, var_name) in enumerate(dynamic.ALL_VARS):

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
        if var_ind < len(dynamic.ALL_VARS)-1:
            plt.setp(this_ax.get_xticklabels(), visible=False)

        # Set the ylabel:
        ylbl = rs_prfs.var_info[PRF_VAL]['prm_name']
        ylbl += f' [{rs_prfs.var_info[PRF_VAL]["prm_unit"]}]'
        this_ax.set_ylabel(dpu.fix_txt(ylbl), labelpad=10)
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
    uncertainties behave like a ucu, ucr, ucs, or uct types or uncertainties.

    Args:
        proc_chunks: the outcome of map(process_chunk, chunks).
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
    for uc_name in [PRF_UCR, PRF_UCS, PRF_UCT, PRF_UCU]:

        # Build matrices of indexes
        i_inds, j_inds = np.meshgrid(np.arange(0, len(covmats[uc_name][0]), 1),
                                     np.arange(0, len(covmats[uc_name][0]), 1))

        # Which position are off-diagonal ?
        off_diag = i_inds != j_inds
        # Out of these, which ones are "valid", i.e. not NaN ?
        valids = off_diag * ~np.isnan(covmats[uc_name])

        # For the uncorrelated uncertainties, all the off-diagonal elements should always be 0
        # (unless they are NaNs). Let's issue a log-ERROR message if this is not the case.
        if uc_name in [PRF_UCR, PRF_UCU]:
            if np.any(covmats[uc_name][valids] != 0):
                logger.error("Non-0 off-diagonal elements of covarience matrix [%s].",
                             uc_name)

        # Now, let's compute the theoretical covariance matrix.
        # This is the covariance of the different elements of the combined profile with itself.
        # As such, it doesn't matter what the mid, rid, eid, oid actaully are - their just the same
        # for all the points in the profile.
        cc_mat = dtgu.coeffs(
            i_inds,  # i
            j_inds,  # j
            uc_name,
            oid_i=np.ones_like(covmats[uc_name]),
            oid_j=np.ones_like(covmats[uc_name]),
            mid_i=np.ones_like(covmats[uc_name]),
            mid_j=np.ones_like(covmats[uc_name]),
            rid_i=np.ones_like(covmats[uc_name]),
            rid_j=np.ones_like(covmats[uc_name]),
            eid_i=np.ones_like(covmats[uc_name]),
            eid_j=np.ones_like(covmats[uc_name]),
            )

        # And now get the uncertainties from the diagonal of the covariance matrix ...
        sigmas = np.atleast_2d(np.sqrt(covmats[uc_name].diagonal()))
        # ... turn them into a masked array ...
        sigmas = np.ma.masked_invalid(sigmas)
        # ... and combine them with the correlation coefficients. Mind the mix of Hadamard and dot
        # products to get the correct mix !
        th_covmats[uc_name] = np.multiply(cc_mat, np.ma.dot(sigmas.T, sigmas))

        # Having done so, we can compute the relative error (in %) that one does by reconstructing
        # the covariance matrix assuming it is exactly of the given ucu/r/s/t type.
        errors[uc_name] = th_covmats[uc_name][valids] / covmats[uc_name][valids] - 1
        errors[uc_name] *= 100

        # If more than 10% of the covariance points have an error greater than 10%, raise a
        # log-error.
        if (tmp := len(errors[uc_name][np.abs(errors[uc_name]) > 10])) > 0.1 * len(valids[valids]):

            msg = 'of the verifiable theoretical covariance matrix elements differ by more than'
            msg = msg + r'10\% of the true value.'
            logger.error(r'Ouch ! %s \% %s', (tmp, msg))

        errors[uc_name] = np.histogram(errors[uc_name],
                                       bins=bins,
                                       density=False)[0] / len(valids[valids]) * 100

        # Finally, store the percentage of covariance elements that can be checked,
        # i.e that were computed.
        perc_covelmts_comp[uc_name] = len(valids[valids]) / (np.size(valids)-len(valids)) * 100

    # Now, let's make a histogram plot of this information
    fig = plt.figure(figsize=(dpu.WIDTH_ONECOL, 5))

    # Use gridspec for a fine control of the figure area.
    fig_gs = gridspec.GridSpec(1, 1,
                               height_ratios=[1], width_ratios=[1],
                               left=0.15, right=0.95,
                               bottom=0.2, top=0.9,
                               wspace=0.05, hspace=0.05)

    ax0 = plt.subplot(fig_gs[0, 0])

    ax0.hist([bins[:-1]]*4, bins, weights=[item[1] for item in errors.items()],
             label=[rf'{item[0]} ({perc_covelmts_comp[item[0]]:.1f}\%)'
                    for item in errors.items()], histtype='step')

    plt.legend()
    ax0.set_xlim((-25, 25))

    ax0.set_xlabel(r'$(V_{i\neq j}^{\rm th}/{V_{i\neq j}}-1)\times 100$')
    ax0.set_ylabel(r'Normalized number count [\%]', labelpad=10)

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


@for_each_var
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
    gdp_filt = tools.get_query_filter(tags_in=gdp_tags+[eid, rid, TAG_GDP],
                                      tags_out=dru.rsid_tags(pop=gdp_tags))
    cws_filt = tools.get_query_filter(tags_in=cws_tags+[eid, rid, TAG_CWS],
                                      tags_out=dru.rsid_tags(pop=cws_tags))

    # Load the GDP profiles
    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(gdp_filt, dynamic.CURRENT_VAR,
                          tdt_abbr=dynamic.INDEXES[PRF_TDT],
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
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
                          ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
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


@for_each_var
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
        raise DvasRecipesError(f'Ouch ! I need a list of mids, not: {mids}')

    # Prepare the search queries
    prf_filt = tools.get_query_filter(tags_in=prf_tags+[eid, rid],
                                      tags_out=dru.rsid_tags(pop=prf_tags)+[TAG_CWS],
                                      mids=mids)

    cws_filt = tools.get_query_filter(tags_in=cws_tags+[eid, rid, TAG_CWS],
                                      tags_out=dru.rsid_tags(pop=cws_tags))

    dta_filt = tools.get_query_filter(tags_in=dta_tags+[eid, rid, TAG_DTA],
                                      tags_out=dru.rsid_tags(pop=dta_tags),
                                      mids=mids)

    # Query these different Profiles
    prfs = MultiProfile()
    prfs.load_from_db(prf_filt, dynamic.CURRENT_VAR, dynamic.INDEXES[PRF_ALT],
                      inplace=True)

    logger.info('Loaded %i Profiles for mids: %s', len(prfs), prfs.get_info('mid'))

    cws_prfs = MultiCWSProfile()
    cws_prfs.load_from_db(cws_filt, dynamic.CURRENT_VAR, dynamic.INDEXES[PRF_TDT],
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                          ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                          uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                          ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                          inplace=True)
    dta_prfs = MultiDeltaProfile()
    dta_prfs.load_from_db(dta_filt, dynamic.CURRENT_VAR,
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                          ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                          uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                          ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                          inplace=True)

    logger.info('Loaded %i DeltaProfiles for mids: %s', len(dta_prfs), dta_prfs.get_info('mid'))

    # Make sure I have the same number of prfs and dta_prfs. Else, it most likely means that some
    # dta were not computed.
    if len(dta_prfs) < len(prfs):
        raise DvasRecipesError('Found less dta_prfs than prfs. ' +
                               'Have all the DeltaProfiles been computed ?')

    # Very well, let's now loop through these, and generate the plot
    for (p_ind, prf) in enumerate(prfs):

        # What is the mid of this prf ?
        mid = prf.info.mid
        logger.info('Processing mid: %s', mid)

        # Extract the corresponding dta profile
        dta = dta_prfs[p_ind]
        # dta_prfs should have the same order as prfs. But let's make sure.
        if dta.info.mid != mid:
            raise DvasRecipesError('"mid" mismatch: wrong sort order between prfs and dta_prfs ?')

        # Let's make some more sanity checks, to avoid bad surprises ...
        if len(cws_prfs) != 1:
            raise DvasRecipesError(f'I got {len(cws_prfs)} != 1 CWS profiles ?!')

        # Let's extract the corresponding SRN and PID
        srn = db_view.srn.values[db_view.oid.values == prfs[p_ind].info.oid][0]
        pid = db_view.pid.values[db_view.oid.values == prfs[p_ind].info.oid][0]

        # Start the plotting
        fig = plt.figure(figsize=(dpu.WIDTH_TWOCOL, 5.5))

        # Create a gridspec structure
        gs_info = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], width_ratios=[1],
                                    left=0.09, right=0.87, bottom=0.12, top=0.93,
                                    wspace=0.5, hspace=0.1)

        # Create the axes - one for the profiles, and one for uctot, ucr, ucs, uct, ucu
        ax0 = fig.add_subplot(gs_info[0, 0])
        ax1 = fig.add_subplot(gs_info[1, 0], sharex=ax0)

        # Extract the DataFrames
        prf_pdf = prfs.get_prms([PRF_ALT, PRF_VAL])[p_ind]
        cws_pdf = cws_prfs.get_prms([PRF_ALT, PRF_VAL, 'uc_tot'])[0]
        dta_pdf = dta_prfs.get_prms([PRF_ALT, PRF_VAL, 'uc_tot'])[p_ind]

        # Show the delta = 0 line
        ax1.plot(cws_pdf.loc[:, PRF_ALT].values, cws_pdf.loc[:, PRF_VAL].values*0,
                 lw=0.4, ls='-', c='darkorchid')

        # Very well, let us plot all these things.
        # First, plot the profiles themselves
        ax0.plot(prf_pdf.loc[:, PRF_ALT].values, prf_pdf.loc[:, PRF_VAL].values,
                 lw=0.4, ls='-', drawstyle='steps-mid', c='k', alpha=1,
                 label=mid[0])
        ax0.plot(cws_pdf.loc[:, PRF_ALT].values, cws_pdf.loc[:, PRF_VAL].values,
                 lw=0.4, ls='-', drawstyle='steps-mid', c='darkorchid', alpha=1, label='CWS')
        ax0.fill_between(prf_pdf.loc[:, PRF_ALT].values,
                         prfs.get_prms(PRF_VAL).max(axis=1).values,
                         prfs.get_prms(PRF_VAL).min(axis=1).values,
                         facecolor=(0.8, 0.8, 0.8), step='mid', edgecolor='none',
                         label='All sondes')

        # Next plot the uncertainties
        ax1.plot(dta_pdf.loc[:, PRF_ALT], dta_pdf.loc[:, PRF_VAL],
                 alpha=1, drawstyle='steps-mid', color='k', lw=0.4)
        ax1.fill_between(dta_pdf.loc[:, PRF_ALT].values,
                         dta_prfs.get_prms(PRF_VAL).max(axis=1).values,
                         dta_prfs.get_prms(PRF_VAL).min(axis=1).values,
                         facecolor=(0.8, 0.8, 0.8), step='mid', edgecolor='none')

        # Display the location of the tropopause and the PBL
        for (loi, symb) in [(MTDTA_TROPOPAUSE, r'$\prec$'), (MTDTA_PBL, r'$\simeq$')]:
            if loi not in prf.info.metadata.keys():
                logger.warning('"%s" not found in CWS metadata.', loi)
                continue

            loi_gph = float(prf.info.metadata[loi].split(' ')[0])

            for ax in [ax0, ax1]:
                ax.axvline(loi_gph, ls=':', lw=1, c='k')
            trans = transforms.blended_transform_factory(ax0.transData, ax0.transAxes)
            ax0.text(loi_gph, 0.95, symb, transform=trans, ha='center', va='top',
                     rotation=90, bbox=dict(boxstyle='square', fc="w", ec="none", pad=0.1))

        # Set the axis labels
        ylbl0 = f'{prfs.var_info[PRF_VAL]["prm_name"]} [{prfs.var_info[PRF_VAL]["prm_unit"]}]'
        ylbl1 = r'$\delta_{e,i}$'
        ylbl1 += f' [{dta_prfs.var_info[PRF_VAL]["prm_unit"]}]'
        altlbl = dta_prfs.var_info[PRF_ALT]['prm_name']
        altlbl += f' [{dta_prfs.var_info[PRF_ALT]["prm_unit"]}]'

        ax0.set_ylabel(dpu.fix_txt(ylbl0), labelpad=10)
        ax1.set_ylabel(dpu.fix_txt(ylbl1), labelpad=10)
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
        dpu.fancy_legend(ax0, '')
        # Add the edt/eid/rid info
        dpu.add_edt_eid_rid(ax0, prfs)

        # Add the source for the plot
        dpu.add_source(fig)

        dpu.add_var_and_k(ax0, mid='+'.join(mid)+rf' \#{pid} ({srn})',
                          var_name=dta_prfs.var_info[PRF_VAL]['prm_name'], k=None)

        # Save it
        fn_suf = dru.fn_suffix(fid=fid, eid=eid, rid=rid, tags=None, mids=mid, pids=[pid],
                               var=dynamic.CURRENT_VAR)
        dpu.fancy_savefig(fig, fn_core='pp', fn_suffix=fn_suf, fn_prefix=dynamic.CURRENT_STEP_ID)


@for_each_var
@log_func_call(logger, time_it=False)
def dtas_per_mid(start_with_tags, mids=None, skip_gdps=False, skip_nongdps=False):
    """ Create a plot of delta profiles grouped by mid.

    Args:
        start_with_tags: which tags to look for in the DB.
        mids (list, optional): list of 'mid' to process. Defaults to None = all
        skip_gdps (bool, optional): if True, any mid with 'GDP' in it will be skipped.
            Defaults to False.
        skip_nongdps (bool, optional): if True, any mid without 'GDP' will be skipped.
    """

    # Format the tags
    tags = dru.format_tags(start_with_tags)

    # Very well, let us first extract the 'mid', if they have not been provided
    db_view = DB.extract_global_view()
    if mids is None:
        mids = db_view.mid.unique().tolist()

    # Basic sanity check of mid
    if not isinstance(mids, list):
        raise DvasRecipesError(f'I need a list of mids, not: {mids}')

    # Very well, let's now loop through these, and generate the plot
    for mid in mids:

        # Second sanity check - make sure the mid is in the DB
        if mid not in db_view.mid.unique().tolist():
            raise DvasRecipesError(f'mid unknown: {mid}')

        # If warranted, skip any GDP profile
        if skip_gdps and '(gdp)' in mid:
            logger.info('Skipping mid: %s', mid)
            continue

        if skip_nongdps and '(gdp)' not in mid:
            logger.info('Skipping mid: %s', mid)
            continue

        # Prepare the search query
        dta_filt = tools.get_query_filter(tags_in=tags+[TAG_DTA],
                                          tags_out=dru.rsid_tags(pop=tags), mids=[mid])

        # Query these DeltaProfiles
        dta_prfs = MultiDeltaProfile()
        dta_prfs.load_from_db(dta_filt, dynamic.CURRENT_VAR,
                              alt_abbr=dynamic.INDEXES[PRF_ALT],
                              ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                              ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                              uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                              ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                              inplace=True)

        # Plot these deltas
        fn_suf = dru.fn_suffix(eid=None, rid=None, tags=tags, mids=[mid], var=dynamic.CURRENT_VAR)
        dpd.dtas(dta_prfs, k_lvl=1, label='mid', show=False,
                 fn_prefix=dynamic.CURRENT_STEP_ID, fn_suffix=fn_suf)
