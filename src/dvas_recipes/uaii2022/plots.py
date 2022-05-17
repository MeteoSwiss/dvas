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
import matplotlib.gridspec as gridspec

# Import dvas modules and classes
from dvas.logger import log_func_call
from dvas.data.data import MultiGDPProfile, MultiCWSProfile, MultiDeltaProfile
from dvas.hardcoded import PRF_REF_INDEX_NAME, PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME
from dvas.hardcoded import PRF_REF_UCU_NAME, PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME
from dvas.hardcoded import TAG_DTA_NAME, TAG_GDP_NAME, TAG_CWS_NAME
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
            database. Defaults to 'sync'.
        label (str, optional): label of the plot legend. Defaults to 'mid'.
        show (bool, optional): if set, overrides the default dvas rule about whether to show the
            plot, or not. Defaults to None.

    """

    # Format the tags
    tags = dru.format_tags(start_with_tags)

    # Extract the flight info
    (eid, rid) = dynamic.CURRENT_FLIGHT

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
        rs_prfs.load_from_db(filt, var_name, dynamic.INDEXES[PRF_REF_TDT_NAME],
                             alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME])
        rs_prfs.sort()  # Sorting is important to make sure I have the same colors for each plot

        # Setup so dummy limits, to keep track of the actual ones as I go.
        xmin = np.infty
        xmax = -np.infty

        # Start looping over all the profiles, and plot them one-by-one.
        for (prf_ind, prf) in enumerate(rs_prfs):

            # Make this plot as a function of the index i, so that synchronized profiles will
            # show up cleanly.
            x = getattr(prf, PRF_REF_VAL_NAME).index.get_level_values(PRF_REF_INDEX_NAME)
            y = getattr(prf, PRF_REF_VAL_NAME).values

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
        ylbl = rs_prfs.var_info[PRF_REF_VAL_NAME]['prm_name']
        ylbl += ' [{}]'.format(rs_prfs.var_info[PRF_REF_VAL_NAME]['prm_unit'])
        this_ax.set_ylabel(dpu.fix_txt(ylbl), labelpad=10)

    # Set the label for the last plot only
    this_ax.set_xlabel(r'$i$')

    # Add the source
    dpu.add_source(fig)

    # Save it all
    dpu.fancy_savefig(fig, 'flight_overview', fn_prefix=dynamic.CURRENT_STEP_ID,
                      fn_suffix=dru.fn_suffix(eid=eid, rid=rid, tags=tags),
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
    for uc_name in [PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME]:

        # Build matrices of indexes
        i_inds, j_inds = np.meshgrid(np.arange(0, len(covmats[uc_name][0]), 1),
                                     np.arange(0, len(covmats[uc_name][0]), 1))

        # Which position are off-diagonal ?
        off_diag = i_inds != j_inds
        # Out of these, which ones are "valid", i.e. not NaN ?
        valids = off_diag * ~np.isnan(covmats[uc_name])

        # For the uncorrelated uncertainties, all the off-diagonal elements should always be 0
        # (unless they are NaNs). Let's issue a log-ERROR message if this is not the case.
        if uc_name in [PRF_REF_UCR_NAME, PRF_REF_UCU_NAME]:
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
            logger.error(rf'Ouch ! {tmp} \% {msg}')

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
             label=[r'{} ({:.1f}\%)'.format(item[0], perc_covelmts_comp[item[0]])
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
    (eid, rid) = dynamic.CURRENT_FLIGHT
    dpu.fancy_savefig(fig, f'covmat_check_chunk-size-{dynamic.CHUNK_SIZE}',
                      fn_prefix=dynamic.CURRENT_STEP_ID,
                      fn_suffix=dru.fn_suffix(eid=eid, rid=rid, var=dynamic.CURRENT_VAR),
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
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    gdp_filt = tools.get_query_filter(tags_in=gdp_tags+[eid, rid, TAG_GDP_NAME],
                                      tags_out=dru.rsid_tags(pop=gdp_tags))
    cws_filt = tools.get_query_filter(tags_in=cws_tags+[eid, rid, TAG_CWS_NAME],
                                      tags_out=dru.rsid_tags(pop=cws_tags))

    # Load the GDP profiles
    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(gdp_filt, dynamic.CURRENT_VAR,
                          tdt_abbr=dynamic.INDEXES[PRF_REF_TDT_NAME],
                          alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME],
                          ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                          ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                          uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                          ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                          inplace=True)
    # Idem for the CWS
    cws_prfs = MultiCWSProfile()
    cws_prfs.load_from_db(cws_filt, dynamic.CURRENT_VAR,
                          tdt_abbr=dynamic.INDEXES[PRF_REF_TDT_NAME],
                          alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME],
                          ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                          ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                          uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                          ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                          inplace=True)

    # We can now create a GDP vs CWS plot ...
    dpg.gdps_vs_cws(gdp_prfs, cws_prfs, show=None,
                    fn_prefix=dynamic.CURRENT_STEP_ID,
                    fn_suffix=dru.fn_suffix(eid=eid, rid=rid, tags=gdp_tags,
                                            var=dynamic.CURRENT_VAR))
    # And a uc_budget plot
    dpg.uc_budget(gdp_prfs, cws_prfs, show=None,
                  fn_prefix=dynamic.CURRENT_STEP_ID,
                  fn_suffix=dru.fn_suffix(eid=eid, rid=rid, tags=gdp_tags,
                                          var=dynamic.CURRENT_VAR))


@for_each_var
@log_func_call(logger, time_it=True)
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
        raise DvasRecipesError('Ouch ! I need a list of mids, not: {}'.format(mids))

    # Very well, let's now loop through these, and generate the plot
    for mid in mids:

        # Second sanity check - make sure the mid is in the DB
        if mid not in db_view.mid.unique().tolist():
            raise DvasRecipesError('Ouch ! mid unknown: {}'.format(mid))

        # If warranted, skip any GDP profile
        if skip_gdps and 'GDP' in mid:
            logger.info('Skipping mid: %s', mid)
            continue

        if skip_nongdps and 'GDP' not in mid:
            logger.info('Skipping mid: %s', mid)
            continue

        # Prepare the search query
        dta_filt = tools.get_query_filter(tags_in=tags+[TAG_DTA_NAME],
                                          tags_out=dru.rsid_tags(pop=tags), mids=[mid])

        # Query these DeltaProfiles
        dta_prfs = MultiDeltaProfile()
        dta_prfs.load_from_db(dta_filt, dynamic.CURRENT_VAR,
                              alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME],
                              ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                              ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                              uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                              ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                              inplace=True)

        # Plot these deltas
        fn_suf = dru.fn_suffix(eid=None, rid=None, tags=tags, mids=[mid], var=dynamic.CURRENT_VAR)
        dpd.dtas(dta_prfs, k_lvl=1, label='mid', show=False,
                 fn_prefix=dynamic.CURRENT_STEP_ID, fn_suffix=fn_suf)
