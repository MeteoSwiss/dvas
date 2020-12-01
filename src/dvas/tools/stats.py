# -*- coding: utf-8 -*-
"""

Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains statistical routines and tools.

"""

# Import from Python packages
import numpy as np
from scipy import stats
import pandas as pd

# Import from this package
from ..dvas_logger import gruan_logger, log_func_call, dvasError
from ..plot import plot_gruan
from .gruan import combine_gdps

# Run a chi-square analysis between a merged profile and its components
@log_func_call(gruan_logger)
def chi_square(gdp_prfs, cws_prf):
    ''' Computes a chi-square given a series of individual profiles, and a merged one.

    This function does not work with resampled profiles: a chi-square is meaningless when computed
    over numerous altitudes at once.

    The chi-square, for n profiles, is computed as::

        1/(n-1) * sum (x_i - <x>)**2/sigma_i**2

    Args:
        gdp_prfs (dvas.data.data.MultiProfile): individual (synchronized!) profiles.
        cws_prf (dvas.dvas.data.MultiProfile): combined (weighted-averaged, binning=1) profile.

    Return:
        ndarray: the chi-square array, with a size of ``len(cws_prf.profiles[0])``.

    '''

    # Let's extract the profile data
    prfs = gdp_prfs.get_prms(['val'])
    cws = cws_prf.get_prms(['val'])[0].values
    cws_uc = cws_prf.get_prms(['uc_tot'])[0].values

    # Make sure no binning was employed.
    for vals in prfs:
        if len(vals) != len(cws):
            raise dvasError('Ouch ! GDP and CWS profiles should have the same length.')

    # Compute the chi square
    chi_sq = np.array([(prf.values - cws)**2/cws_uc**2 for prf in prfs])
    chi_sq *= (len(prfs)-1)**-1

    # Use np.where to return nan (instead of 0) when this is all I have
    chi_sq = np.where(np.all(np.isnan(chi_sq), axis=0), np.nan, np.nansum(chi_sq, axis=0))

    return chi_sq


@log_func_call(gruan_logger)
def ks_test(gdp_pair, alpha=0.0027, binning_list=None, do_plot=False, **kwargs):
    ''' Runs a "rolling" KS test between two columns of a pandas.DataFrame.

    The rolling refers to the KS test being executed on a series of binned versions of the column
    deltas.

    Args:
        vals (pd.DataFrame): 2-column DataFrame with the data
        uc_tots (pd.DataFrame): 2-column DataFrame with the total uncertainties, with the same shape
            as vals.
        alpha (float or list, optional): The significance level for the KS test. Defaults to 0.27%
        binning_list (ndarray of int, optional): The rolling binning sizes. Defaults to [1].
        do_plot (bool, optional): Whether to create the diagnostic plot, or not. Defaults to False.
        **kwargs: will be fed to the underlying plot function.

    Returns:
        ndarray of int: The array of flags with shape (len(binning_list), len(vals)), with 1's where
        the KS test failed, and 0 otherwise. That is, 1 <=> the p-value of the KS test is <= alpha.

    Note:
        The diagnostic plot will be generated only if a binning of 1 is included in the
        binning_list.

    Todo:
        * When rolling, take into account the previously flagged data point.
        * Deal with the plot tag.

    '''

    # Some sanity checks
    if binning_list is None:
        binning_list = [1]

    if not isinstance(binning_list, list):
        raise Exception('Ouch! binning_list should be a list, not %s' % (type(binning_list)))

    if any([not isinstance(val, np.int) for val in binning_list]):
        raise Exception('Ouch! binning_list must be a list of int, not %s' % (np.str(binning_list)))

    if not isinstance(alpha, np.float):
        raise Exception('Ouch! alpha should be a float, not %s.' % (type(alpha)))

    if alpha > 1 or alpha < 0:
        raise Exception('Ouch! alpha should be 0<alpha<1, not %.1f.' % (alpha))

    # How long are the profiles ?
    len_prf = vals.shape[0]

    # Let's create a flag array, and also one for storing the p_values (for a plot).
    f_pqi = np.zeros((len(binning_list), len_prf))
    p_ksi = np.zeros_like(f_pqi)

    # Let's start rolling
    k_pqis = []

    for (b_ind, binning) in enumerate(binning_list):

        # Compute the profile delta with the specified sampling
        gdp_delta = combine_gdps(gdp_pair, binning=binning, method='delta')

        # Compute k_pqi (the normalized profile delta)
        k_pqi = gdp_delta.profiles[0].values/gdp_delta.get_prms('uc_tot')[0].values
        k_pqis += [k_pqi]

        # Loop through each level and run a KS test on it.
        for k_ind in range(len(k_pqi)):

            # Get the p-value of the KS test
            p_val = stats.kstest(np.array([k_pqi[k_ind]]), 'norm', args=(0, 1)).pvalue

            # Store it
            p_ksi[b_ind][prof_del_old_inds[k_ind][0] : prof_del_old_inds[k_ind][-1]+1] = p_val

    # Compute the flags.
    f_pqi[p_ksi <= alpha] = 1

    # Create a diagnostic plot. I can do this only if one of the binning values is 1.
    if 1 in binning_list and do_plot:
        plot_gruan.plot_ks_test(k_pqis[binning_list.index(1)], f_pqi, p_ksi, binning_list, alpha,
                                tag='')
    else:
        gruan_logger.warning('KS test ran without binning of 1. Skipping the diagnostic plot.')

    return (f_pqi, p_ksi)

@log_func_call(gruan_logger)
def check_gdp_compatibility(gdp_prfs, alpha=0.0027, binning_list=None, do_plot=False, **kwargs):
    ''' Runs a series of KS test to assess the consistency of several GDP profiles.

    Args:
        gdp_profs (dvas.data.data.MultiGDPProfile): synchronized GDP profiles to check.
        alpha (float, optional): The significance level for the KS test. Defaults to 0.27%
        binning_list (ndarray of int, optional): The rolling binning sizes. Defaults to [1].
        do_plot (bool, optional): Whether to create the diagnostic plot, or not. Defaults to False.

    Returns:
        ndarray: The array of flags of len(binning_list), with 1's where the KS test failed.
        That is, where the p-value of the KS test is <= alpha.

    Note:
        The diagnostic plot will be generated only if a binning of 1 is included in the
        binning_list.

    Todo:
        * When rolling, take into account the previously flagged data point.
        * Deal with the plot tag.

    '''


    # How many gdp Proficles do I have ?
    n_prf = len(gdp_prfs)
    # How many KS tests do I need to make ? (They are symetric)
    n_test = (n_prf-1)*n_prf//2

    # What the are the indices of the profile pairs I want to check. Make sure to check each pair
    # only once.
    prf_a_inds = [[i]*(n_prf-1-i) for i in range(n_prf-1)]
    prf_a_inds = [item for sublist in prf_a_inds for item in sublist]
    prf_b_inds = [[i for i in range(k, n_prf)] for k in range(1, n_prf)]
    prf_b_inds = [item for sublist in prf_b_inds for item in sublist]

    # I'll want to keep track of the compatibility of each pofile pairs
    f_pqi = []

    # Let us loop through all these test and run them sequentially.
    for test_id in range(n_test):

        # Exctract the specific profile pair I need to assess
        gdp_pair = gdp_prfs.extract([prf_a_inds[test_id], prf_b_inds[test_id]])

        # Run the KS test on it
        ks_test(gdp_pair, alpha=alpha, binning_list=binning_list, do_plot=do_plot, **kwargs)
