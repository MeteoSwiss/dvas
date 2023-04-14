# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains statistical routines and tools for handling GDPs.

"""

# Import from Python packages
import logging
import numbers
import numpy as np
import pandas as pd
from scipy import stats

# Import from this package
from ...logger import log_func_call
from ...errors import DvasError
from .gdps import combine
from ...plots import gdps as dpg
from ...plots import utils as dpu
from ...hardcoded import PRF_VAL, FLG_INCOMPATIBLE, FLG_ISINVALID

# Setup local logger
logger = logging.getLogger(__name__)


# @log_func_call(logger, time_it=True)
# def chi_square(gdp_prfs, cws_prf):
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
def ks_test(gdp_pair, alpha=0.0027, m_val=1, method='arithmetic delta', **kwargs):
    ''' Runs a ``scipy.stats.kstest()`` two-sided test on the normalized-delta between two
    GDPProfile instances, against a normal distribution.

    The KS test is being run on a level-per-level basis.

    Note:

        See the dvas documentation for more details about the scientific motivation for this
        function.

    Args:
        gdp_pair (list of dvas.data.strategy.data.GDPProfile): GDP Profiles to compare.
        alpha (float, optional): The significance level for the KS test. Must be 0<alpha<1.
            Defaults to 0.27%=0.0027.
        m_val (int, optional): Binning strength for the Profile delta (Important: the binning is
            performed before running the KS test). Defaults to 1 (=no binning).
        method (str, optional): 'arithmetic delta' or 'circular delta' (the latter wraps angles)
        **kwargs: mask_flgs and/or n_cpus and/or chunk_size, that will get fed to
            dvas.tools.gdps.gdps.combine().

    Returns:
        pandas DataFrame: a DataFrame containing Delta_pqei, sigma_pqei, k_pqei, pks_pqei, and
        f_pqei values:

            * Delta_pqei is the (binned) profile delta with sigma_pqei its (total) uncertainty,

            * k_pqei=Delta_pqei/sigma_pqei is the (binned) profile delta normalized by the total
              uncertainty,

            * pks_pqei contains the corresponding p-value from the KS test, and

            * f_pqi contains 1 where the KS test failed, and 0 otherwise.
              That is: 1 <=> the p-value of the KS test is <= alpha.

    '''

    if not isinstance(m_val, numbers.Integral):
        raise DvasError(f'binning should be an int, not {type(m_val)}')

    if not isinstance(alpha, float):
        raise Exception(f'alpha should be a float, not {type(alpha)}')

    if alpha > 1 or alpha < 0:
        raise Exception(f'alpha should be 0<alpha<1, not {alpha}')

    # How long are the profiles ?
    # len_prf = len(gdp_pair[0].data)

    if (tmp1 := len(gdp_pair[1])) != (tmp0 := len(gdp_pair[0])):
        raise DvasError(f"GDP Profiles have inconsistent lengths: {tmp0} vs {tmp1}")

    # Compute the profile delta with the specified sampling
    gdp_delta, _ = combine(gdp_pair, binning=m_val, method=method, **kwargs)

    # Let's create a DataFrame to keep track of incompatibilities.
    out = pd.DataFrame(np.full((len(gdp_delta[0]), 5), np.nan),
                       columns=['Delta_pqei', 'sigma_pqei', 'k_pqei', 'f_pqei', 'pks_pqei'])

    # Assign the first part of the data to it
    out[['Delta_pqei', 'sigma_pqei']] = gdp_delta.get_prms([PRF_VAL, 'uc_tot'])[0]

    # Compute k_pqei
    out['k_pqei'] = out['Delta_pqei']/out['sigma_pqei']

    # Loop through each level and run a KS test on it.
    out['pks_pqei'] = [stats.kstest(np.array([k_val]), 'norm', args=(0, 1)).pvalue
                       for k_val in out['k_pqei'].values]

    # Compute the flags.
    # The following two lines may seem redundant, but it'll make sure NaN's get a NaN flag.
    out.loc[out['pks_pqei'] <= alpha, 'f_pqei'] = 1
    out.loc[out['pks_pqei'] > alpha, 'f_pqei'] = 0
    # Set the proper dtype
    out['f_pqei'] = out['f_pqei'].astype('Int64')

    return out


@log_func_call(logger, time_it=True)
def gdp_incompatibilities(gdp_prfs, alpha=0.0027, m_vals=None, method='arithmetic delta',
                          rolling=True, do_plot=False, fn_prefix=None, fn_suffix=None,
                          **kwargs):
    ''' Runs a series of KS tests to assess the consistency of several GDP profiles.

    Args:
        gdp_prfs (dvas.data.data.MultiGDPProfile): synchronized GDP profiles to check.
        alpha (float, optional): The significance level for the KS test. Defaults to 0.27%
        m_vals (ndarray of int, optional): The rolling binning sizes "m". Defaults to None==[1].
        method (str, optional): 'arithmetic delta' or 'circular delta' (the latter wraps angles).
        rolling (bool, optional): if True and len(m_vals)>1, any incompatibility found
            for a specific m value will be forwarded to the subsequent ones. Else, each m value
            is treated independantly. Defaults to True. If rolling is True, the order of the
            m_vals list thus matters.
        do_plot (bool, optional): Whether to create the diagnostic plot, or not. Defaults to False.
        fn_prefix (str, optional): if set, the prefix of the plot filename.
        fn_suffix (str, optional): if set, the suffix of the plot filename.
        **kwargs: n_cpus and/or chunk_size and/or mask_flgs, that will get fed to
            dvas.tools.gdps.gdps.combine().

    Returns:
        pd.DataFrame: the values of Delta_pqei, sigma_pqei, k_pqei, pks_pqei and f_pqei for each
        pair of GDPs and each m value.

        GDP pairs are identified using their oids, as: `oid_1_vs_oid_2`.

        f_pqei==1 indicates that the p-value of the KS test is <= alpha for this measurement,
        i.e. that the profiles are incompatible.

    '''

    # Some sanity checks to begin with
    if m_vals is None:
        m_vals = [1]

    # Be extra courteous if someone gives me an int
    if isinstance(m_vals, int):
        m_vals = [m_vals]
    if not isinstance(m_vals, list):
        raise DvasError(f'm_vals must be a list, not: {type(m_vals)}')

    # If warranted select which flags we want to mask in the rolling process.
    mask_flgs = [FLG_ISINVALID]
    if rolling:
        mask_flgs += [FLG_INCOMPATIBLE]

    # How many gdp Profiles do I have ?
    n_prf = len(gdp_prfs)

    # How many KS tests do I need to make ? (They are symmetric)
    n_test = ((n_prf-1)*n_prf)//2

    # What are the indices of the profile pairs I want to check ? Make sure to check each pair
    # only once.
    prf_a_inds = [[i]*(n_prf-1-i) for i in range(n_prf-1)]
    prf_a_inds = [item for sublist in prf_a_inds for item in sublist]
    prf_b_inds = [range(k, n_prf) for k in range(1, n_prf)]
    prf_b_inds = [item for sublist in prf_b_inds for item in sublist]

    # Prepare a dictionnary to hold the results
    incompat = {}

    # Let us loop through all these tests and run them sequentially.
    for test_id in range(n_test):

        # Extract the specific profile pair I need to assess
        gdp_pair = gdp_prfs.extract([prf_a_inds[test_id], prf_b_inds[test_id]])

        # Get the oids to form the dictionary key
        key = '_vs_'.join([str(item.info.oid) for item in gdp_pair])
        logger.debug('GPD oids: %s', key)

        # First make a high-resolution delta ...
        out = ks_test(gdp_pair, alpha=alpha, m_val=1, mask_flgs=mask_flgs, method=method, **kwargs)

        # Drop un-necessary info fromthe High-Kes k_pqei profile
        out = out.drop(labels=['pks_pqei', 'f_pqei'], axis=1)

        # Turn this into a DataFrame with MultiIndex to store things that are coming
        # Lots of index swapping here, until I get things right ...
        # out = pd.DataFrame(out, columns=['k_pqi'])
        out = out.reset_index(drop=True)
        out = pd.DataFrame(out.values,
                           columns=pd.MultiIndex.from_tuples([(0, item) for item in out.columns],
                                                             names=('m', None)))

        # Get started with the rolling KS test
        for m_val in m_vals:

            # Run the KS test on it
            tmp = ks_test(gdp_pair, alpha=alpha, m_val=m_val, mask_flgs=mask_flgs, method=method,
                          **kwargs)

            # Turn this into a MultiIndex ...
            tmp.columns = pd.MultiIndex.from_tuples([(m_val, item) for item in tmp.columns])

            # If binning >1, this will have the wrong size.
            # Correct it (i.e. tie the KS results back to the original levels) ...
            # Inspired from the answer of DSM on Stack Overflow:
            # https://stackoverflow.com/questions/26777832/
            tmp = tmp.reset_index(drop=True)
            tmp = tmp.loc[np.repeat(range(len(tmp)), m_val)]
            tmp = tmp.reset_index(drop=True)
            tmp = tmp[:len(gdp_pair[0])]

            # ... and append it to the rest of the data for this pair.
            out = pd.concat([out, tmp], axis=1)

            # If requested, flag the bad levels detected using this binning intensity
            if rolling:
                for gdp in gdp_pair:
                    gdp.set_flg(FLG_INCOMPATIBLE, True,
                                index=out[out[(m_val, 'f_pqei')] == 1].index)

        # Assign this pair's outcome to the final storage dictionnary
        incompat[key] = out

        # Plot things if needed
        if do_plot:

            # Is there a unit for the data at hand ?
            try:
                var_name = gdp_pair.var_info[PRF_VAL]['prm_plot']
                var_unit = gdp_pair.var_info[PRF_VAL]['prm_unit']
            except KeyError:
                var_name = None
                var_unit = None

            # Extract the edt eid rid info for the pair
            edt_eid_rid_info = dpu.get_edt_eid_rid(gdp_pair)

            # Get the specific pair details
            # Start with the second profile, since the delta does Profile2-Profile1, but
            # people usually understand the opposite when writing Profile1_vs_Profile2.
            pair_info = f'{"-".join([f"{item}" for item in gdp_pair[1].info.mid])}_'
            pair_info += f'[{"-".join([f"{item}" for item in gdp_pair[1].info.oid])}]_minus_'
            pair_info += f'{"-".join([f"{item}" for item in gdp_pair[0].info.mid])}_'
            pair_info += f'[{"-".join([f"{item}" for item in gdp_pair[0].info.oid])}]'

            fnsuf = pair_info
            if fn_suffix is not None:
                fnsuf = fn_suffix + '_' + fnsuf

            dpg.plot_ks_test(out, alpha,
                             left_label=edt_eid_rid_info+' '+pair_info.replace('_', ' '),
                             right_label=var_name, unit=var_unit,
                             fn_prefix=fn_prefix, fn_suffix=fnsuf)

    # Here, get rid of the dictionnary and group everything under a single DataFrame
    return pd.concat(incompat, axis=1, names=['gdp_pair', 'm', None])


@log_func_call(logger, time_it=True)
def gdp_validities(incompat, m_vals=None, strategy='all-or-none'):
    """ Given GDP incompatibilities, identifies valid measurements (suitable for the assembly of a
    combined working standard) given a specific combination strategy.

    Valid strategies include:
        * 'all-or-none': either all GDP measurements from a certain bin are compatible with each
            others, or all of them are dropped.
        * 'force-all-valid': combine all GDPs, irrespective of the reported incompatibilities.

    Args:
        incompat (dict): outcome of dvas.tools.gdps.stats.gdp_incompatibilities().
        m_vals (list of int, oprtional): list of m values to take into account when checking
            incompatibilities. Defaults to None = [1].
        strategy (str, optional): name of a validation strategy. Defaults to 'all-or-none'.

    Returns:

    """

    # Begin with some sanoty checks
    if m_vals is None:
        m_vals = [1]
    if isinstance(m_vals, int):
        m_vals = [m_vals]
    if np.any([not isinstance(val, int) for val in m_vals]):
        raise DvasError(f'Ouch ! m_vals should be a list of int, not: {m_vals}')

    # How many profiles are being validated ?
    # Note: this is inverting the equation 1/2*n*(n-1)=t, with t the total number of (unique)
    # comparisons between n gdp profiles.
    n_gdps = int(0.5 + np.sqrt(2*len(incompat.columns.unique('gdp_pair')) + 0.25))

    # What are the oids of these gdps ?
    oids = [oid for key in incompat.columns.unique('gdp_pair') for oid in key.split('_vs_')]
    oids = sorted(set(oids))

    # Quick sanity check to make sure I did not mess anything up
    assert len(oids) == n_gdps

    # Let's prepare the structure in which we will list which measurement is "valid" and which one
    # isn't. This is a DataFrame with the oid set as the column names.
    valids = pd.DataFrame(index=range(len(incompat)), columns=oids)

    # Extract the values that actually matter for checking the validitiy regions, i.e. f_pqi for
    # all the chosen m values.
    incompat = incompat.iloc[:,
                             (incompat.columns.get_level_values(2) == 'f_pqei') *
                             (incompat.columns.get_level_values('m').isin(m_vals))]

    # The easiest (and most restrictive) validation strategy: a level is valid only if all
    # GDPs are compatible with each others at that level. Else, drop the level entirely.
    if strategy == 'all-or-none':

        # Here, all profiles will have the same validities (i.e. either all ok, or None ok).
        for oid in oids:
            valids[oid] = incompat.sum(axis=1, skipna=False) == 0  # valid if NO incompatibilities.
    elif strategy == 'force-all-valid':
        for oid in oids:
            valids[oid] = True
    else:
        raise DvasError(f'Ouch ! Unknown validation strategy: {strategy}')

    return valids
