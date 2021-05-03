# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains statistical routines and tools for handling GDPs.

"""

# Import from Python packages
import numpy as np
import pandas as pd
from scipy import stats

# Import from this package
from ...logger import log_func_call
from ...logger import tools_logger as logger
from ...errors import DvasError
from .gdps import combine
from ...plots import gdps as dpg
from ...hardcoded import PRF_REF_VAL_NAME

#@log_func_call(logger, time_it=True)
#def chi_square(gdp_prfs, cws_prf):
#    ''' Computes a chi-square given a series of individual profiles, and a merged one.
#
#    This function does not work with resampled profiles: a chi-square is meaningless when computed
#    over numerous altitudes at once.
#
#    The chi-square, for n profiles, is computed as::
#
#        1/(n-1) * sum (x_i - <x>)**2/sigma_i**2
#
#    Args:
#        gdp_prfs (dvas.data.data.MultiProfile): individual (synchronized!) profiles.
#        cws_prf (dvas.dvas.data.MultiProfile): combined (weighted-averaged, binning=1) profile.
#
#    Return:
#        ndarray: the chi-square array, with a size of ``len(cws_prf.profiles[0])``.
#
#    '''
#
#    # Let's extract the profile data
#    prfs = gdp_prfs.get_prms(['val'])
#    cws = cws_prf.get_prms(['val'])[0].values
#    cws_uc = cws_prf.get_prms(['uc_tot'])[0].values
#
#    # Make sure no binning was employed.
#    for vals in prfs:
#        if len(vals) != len(cws):
#            raise DvasError('Ouch ! GDP and CWS profiles should have the same length.')
#
#    # Compute the chi square
#    chi_sq = np.array([(prf.values - cws)**2/cws_uc**2 for prf in prfs])
#    chi_sq *= (len(prfs)-1)**-1
#
#    # Use np.where to return nan (instead of 0) when this is all I have
#    chi_sq = np.where(np.all(np.isnan(chi_sq), axis=0), np.nan, np.nansum(chi_sq, axis=0))
#
#    return chi_sq

@log_func_call(logger, time_it=True)
def ks_test(gdp_pair, alpha=0.0027, binning=1, n_cpus=1):
    ''' Runs a ``scipy.stats.kstest()`` two-sided test on the normalized-delta between two
    GDPProfile instances, against a normal distribution.

    The KS test is being run on a level-pre-level basis.

    Note:

        See the dvas documentation for more details about the scientific motivation for this
        function.

    Args:
        gdp_pair (list of dvas.data.strategy.data.GDPProfile): GDP Profiles to compare.
        alpha (float, optional): The significance level for the KS test. Must be 0<alpha<1.
            Defaults to 0.27%.
        binning (int, optional): Whether to bin the Profile delta before running the KS test.
            Defaults to 1 (=no binning).
        n_cpus (int|str, optional): number of cpus to use. Can be a number, or 'max'. Set to 1 to
            disable multiprocessing. Defaults to 1.

    Returns:
        pandas DataFrame: a DataFrame containing k_pqi, p_ksi, and f_pqi values. k_pqi contains the
        (binned) profile delta normalized by the total uncertainty. p_ksi contains the
        corresponding p-value from the KS test. f_pqi contains 1 where the KS test failed, and
        0 otherwise. That is: 1 <=> the p-value of the KS test is <= alpha.
    '''

    if not isinstance(binning, int):
        raise DvasError('Ouch! binning should be an int, not %s' % (type(binning)))

    if not isinstance(alpha, float):
        raise Exception('Ouch! alpha should be a float, not %s.' % (type(alpha)))

    if alpha > 1 or alpha < 0:
        raise Exception('Ouch! alpha should be 0<alpha<1, not %.1f.' % (alpha))

    # How long are the profiles ?
    len_prf = len(gdp_pair[0].data)

    if (tmp := len(gdp_pair[1].data)) != len_prf:
        raise DvasError("Ouch ! GDP Profiles have inconsistent lengths: {} vs {} ".format(len_prf,
                                                                                          tmp))

    # Let's create a DataFrame to keep track of incompatibilities.
    out = pd.DataFrame(np.zeros((len_prf, 3)), columns=['k_pqi', 'f_pqi', 'p_ksi'])

    # Compute the profile delta with the specified sampling
    gdp_delta = combine(gdp_pair, binning=binning, method='delta', n_cpus=n_cpus)

    # Compute k_pqi (the normalized profile delta)
    k_pqi = gdp_delta.get_prms([PRF_REF_VAL_NAME, 'uc_tot'])[0]
    k_pqi = k_pqi[PRF_REF_VAL_NAME]/k_pqi['uc_tot']

    # Setup a DataFrame that will eventually get out of this function
    out = pd.DataFrame(k_pqi, columns=['k_pqi'])

    # Loop through each level and run a KS test on it.
    out['p_ksi'] = [stats.kstest(np.array([k_val]), 'norm', args=(0, 1)).pvalue for k_val in k_pqi]

    # Compute the flags.
    # The following two lines may seem redundant, but it'll make sure NaN's get a NaN flag.
    out.loc[out['p_ksi'] <= alpha, 'f_pqi'] = 1
    out.loc[out['p_ksi'] > alpha, 'f_pqi'] = 0
    # Set the proper dtype
    out['f_pqi'] = out['f_pqi'].astype('Int64')

    return out

@log_func_call(logger, time_it=True)
def get_incompatibility(gdp_prfs, alpha=0.0027, bin_sizes=None, rolling_flags=True,
                        do_plot=False, n_cpus=1, **kwargs):
    ''' Runs a series of KS tests to assess the consistency of several GDP profiles.

    Args:
        gdp_profs (dvas.data.data.MultiGDPProfile): synchronized GDP profiles to check.
        alpha (float, optional): The significance level for the KS test. Defaults to 0.27%
        bin_sizes (ndarray of int, optional): The rolling binning sizes. Defaults to [1].
        rolling_flags (bool, optional):
        do_plot (bool, optional): Whether to create the diagnostic plot, or not. Defaults to False.
        n_cpus (int|str, optional): number of cpus to use. Can be a number, or 'max'. Set to 1 to
            disable multiprocessing. Defaults to 1.

    Returns:
        dict of ndarray of bool: list of pair-wise incompatible measurements between GDPs.
            Each distionary entry is labeled: "oid_1__vs__oid_2", to identify the profile pair.
            True  indicates that the p-value of the KS test is <= alpha for a specific measurement.

    Note:
        The diagnostic plot will be generated only if a binning of 1 is included in the
        binning_list.

    Todo:
        * When rolling, take into account the previously flagged data point.
        * Deal with the plot tag.

    '''

    # Some sanity checks to begin with
    if bin_sizes is None:
        bin_sizes = [1]

    # Be extra courteous if someone gives me an int
    if isinstance(bin_sizes, int):
        bin_sizes = list(bin_sizes)

    if not isinstance(bin_sizes, list):
        raise DvasError('Ouch ! bin_sizes must be a list, not: {}'.format(type(bin_sizes)))

    # How many gdp Profiles do I have ?
    n_prf = len(gdp_prfs)

    # How many KS tests do I need to make ? (They are symmetric)
    n_test = ((n_prf-1)*n_prf)//2

    # What the are the indices of the profile pairs I want to check. Make sure to check each pair
    # only once.
    prf_a_inds = [[i]*(n_prf-1-i) for i in range(n_prf-1)]
    prf_a_inds = [item for sublist in prf_a_inds for item in sublist]
    prf_b_inds = [range(k, n_prf) for k in range(1, n_prf)]
    prf_b_inds = [item for sublist in prf_b_inds for item in sublist]

    # Prepare a dictionnary to hold the results
    incompat = {}

    # Let us loop through all these test and run them sequentially.
    for test_id in range(n_test):

        # Extract the specific profile pair I need to assess
        gdp_pair = gdp_prfs.extract([prf_a_inds[test_id], prf_b_inds[test_id]])

        # Get the oids to form the dictionary key
        key = '__vs__'.join([str(item.info.oid) for item in gdp_pair])

        # First make a high-resolution delta ...
        out = combine(gdp_pair, binning=1, method='delta', n_cpus=n_cpus)

        # ... and extract the DataFrame Compute k_pqi (the normalized profile delta)
        out = out.get_prms([PRF_REF_VAL_NAME, 'uc_tot'])[0]
        out = out[PRF_REF_VAL_NAME]/out['uc_tot']

        # Turn this into a DataFrame with MultiIndex to store things that are coming
        # Lots of index swapping here, until I get things right ...
        out = pd.DataFrame(out, columns=['k_pqi'])
        out = out.reset_index(drop=False)
        out = pd.DataFrame(out.values,
                           columns=pd.MultiIndex.from_tuples([(0, item) for item in out.columns],
                                                             names=('m', None)))

        # Get started with the rolling KS test
        for binning in bin_sizes:

            # Run the KS test on it
            tmp = ks_test(gdp_pair, alpha=alpha, binning=binning)

            # Turn this into a MultiIndex ...
            tmp.columns = pd.MultiIndex.from_tuples([(binning, item) for item in tmp.columns])

            # If binning >1, this will have the wrong size. Correct it (i.e. tie the KS results back
            # to the original levels) ...
            # Inspired from the asnwer of DSM on this following Stack Overflow post:
            # https://stackoverflow.com/questions/26777832/replicating-rows-in-a-pandas-data-frame-by-a-column-value
            tmp = tmp.reset_index(drop=True)
            tmp = tmp.loc[np.repeat(range(len(tmp)), binning)]
            tmp = tmp.reset_index(drop=True)
            tmp = tmp[:len(gdp_pair[0])]

            # ... and append it to the rest of the data for this pair.
            out = pd.concat([out, tmp], axis=1)

            # If requested, flag the bad levels detected using this binning intensity
            if rolling_flags:
                for gdp in gdp_pair:
                    gdp.set_flg('incomp', True, index=out[out[(binning, 'f_pqi')] == 1].index)

        # Assign this pair's outcome to the final storage dictionnary
        incompat[key] = out

        # Plot things if needed
        if do_plot:
            dpg.plot_ks_test(out, alpha, title=key, fn_suffix=key)

    return incompat
