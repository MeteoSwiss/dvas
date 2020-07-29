# -*- coding: utf-8 -*-
"""

Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains GRUAN-related stats routines.

"""

# Import from Python packages
import numpy as np
from scipy import stats

# Import from this package
from . import gruan
from ..dvas_logger import gruan_logger, log_func_call


@log_func_call(gruan_logger)
def gdp_ks_test(profiles, sigma_us, sigma_es, sigma_ss, sigma_ts,
                alpha = 0.0027, binning_list=None, **kwargs):
    ''' Runs a KS test to assess the consistency between 2 GDP profiles.

    Args:
        profiles (list of ndarray): list of profiles to combine. All must have the same length!
        sigma_us (list of ndarray): list of associated uncorrelated errors.
        sigma_es (list of ndarray): list of associated environmental-correlated errors.
        sigma_ss (list of ndarray): list of associated spatial-correlated errors.
        sigma_ts (list of ndarray): list of associated temporal-correlated errors.
        alpha (float, optional): The significance level for the KS test. Default: 0.27%
        binning_list (ndarray of int): The rolling binning sizes.

    Returns:
        ndarray: The array of flags of len(binning_list), with 1's where the KS test failed.

    Todo:
        * Improve the input format.

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
    n_steps = len(profiles[0])

    # Let's create a flag array, and also one for storing the p_values (for a plot).
    f_pqi = np.zeros((len(binning_list), n_steps))
    p_ksi = np.zeros_like(f_pqi)

    # Let's start rolling
    k_pqis = []
    for (b_ind, binning) in enumerate(binning_list):

        # Compute the profile delta with the specified sampling
        # TODO: account for previously flagged bad data points
        (prof_del, prof_del_sigma_u, prof_del_sigma_e, prof_del_sigma_s, prof_del_sigma_t,
         prof_del_old_inds, _) = \
         gruan.merge_andor_rebin_gdps(profiles, sigma_us, sigma_es, sigma_ss, sigma_ts,
                                      binning=binning, method='delta', **kwargs)

        # Extract the variance of the difference
        delta_pqi = prof_del
        vu_delta_pqi = prof_del_sigma_u**2
        ve_delta_pqi = prof_del_sigma_e**2
        vs_delta_pqi = prof_del_sigma_s**2
        vt_delta_pqi = prof_del_sigma_t**2

        # Compute the total variance
        v_delta_pqi = vu_delta_pqi + ve_delta_pqi + vs_delta_pqi + vt_delta_pqi

        # Compute k_pqi (the normalized profile delta)
        k_pqi = delta_pqi/np.sqrt(v_delta_pqi)
        k_pqis += [k_pqi]

        # Loop through each level and run a KS test on it.
        for k_ind in range(len(k_pqi)):

            # Get the p-value of the KS test
            p_val = stats.kstest(np.array([k_pqi[k_ind]]), 'norm', args=(0, 1)).pvalue

            # Store it
            p_ksi[b_ind][prof_del_old_inds[k_ind][0] : prof_del_old_inds[k_ind][-1]+1] = p_val

    # Compute the flags.
    f_pqi[p_ksi < alpha] = 1

    return (f_pqi, p_ksi)

# Run a chi-square analysis between a merged profile and its components
def chi_square(profiles, sigma_us, sigma_es, sigma_ss, sigma_ts,
               profile_m, sigma_u_m, sigma_e_m, sigma_s_m, sigma_t_m):
    ''' Computes a chi-square given a series of individual profiles, and a merged one.

    This function does not work with resampled profiles: a chi-square is meaningless when computed
    over numerous altitudes at once.

    The chi-square, for n profiles, is computed as::

        1/(n-1) * sum(x_i - <x>)**2/sigma_i**2

    Args:
        profiles (list of ndarray): list of individual profiles. All must have the same length!
        sigma_us (list of ndarray): list of associated uncorrelated errors.
        sigma_es (list of ndarray): list of associated environmental-correlated errors.
        sigma_ss (list of ndarray): list of associated spatial-correlated errors.
        sigma_ts (list of ndarray): list of associated temporal-correlated errors.
        profile_m (ndarray): Mean profile. Must have the same length!
        sigma_u_m (ndarray): list of associated uncorrelated errors.
        sigma_e_m (ndarray): list of associated environmental-correlated errors.
        sigma_s_m (ndarray): list of associated spatial-correlated errors.
        sigma_t_m (ndarray): list of associated temporal-correlated errors.

    Return:
        ndarray: the chi-square array, with a size of ``len(profile_m)``.

    TODO:
        - implement dedicated test(s)

    '''

    # Figure out the common length
    n_steps = len(profile_m)

    for profile in profiles:
        if len(profile) != n_steps:
            raise Exception('Ouch ! All profiles should have the same length.')

    # Compute the chi square
    chi_sq = np.array([(profiles[ind] - profile_m)**2/(sigma_us[ind]**2 + sigma_es[ind]**2 + sigma_ss[ind]**2 + sigma_ts[ind]**2) for ind in range(len(profiles))])
    chi_sq *= (len(profiles)-1)**-1

    # Use np.where to return nan (instead of 0) when this is all I have
    chi_sq = np.where(np.all(np.isnan(chi_sq), axis=0), np.nan, np.nansum(chi_sq, axis=0))

    return chi_sq
