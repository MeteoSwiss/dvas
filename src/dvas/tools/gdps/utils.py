# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains GRUAN-related utilities.

"""

# WARNING: this package should NOT import anything from dvas.data.
# The routines included must be designed in a generci manner, such that they can also be
# exploited by MultiProfile Strategies without any recursive import problems.

# TODO: is there a better way to do this ?

# Import from Python
import numpy as np
import pandas as pd

# Import from current package
from ...logger import log_func_call
from ...logger import tools_logger as logger
from ...errors import DvasError
from ...hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME, PRF_REF_FLG_NAME
from ...hardcoded import PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME
from ..tools import fancy_nansum


@log_func_call(logger)
def corcoefs(i, j, sigma_name, oid_i=None, oid_j=None, mid_i=None, mid_j=None,
             rid_i=None, rid_j=None, eid_i=None, eid_j=None):
    ''' Computes the correlation coefficient(s), for the different uncertainty types of GRUAN Data
    Products (GDPs), between specific measurements.

    Args:
        i (numpy.ndarray of int|float): synchronized time, step, or altitude of measurement 1.
        j (numpy.ndarray of int|float): synchronized time, step, or altitude of measurement 2.
        sigma_name (str): uncertainty type. Must be one of ['ucr', 'ucs', 'uct', 'ucu'].
        oid_i (numpy.ndarray of int|str, optional): object id from measurement 1.
        oid_j (numpy.ndarray of int|str, optional): object id from measurement 2.
        mid_i (numpy.ndarray of int|str, optional): GDP model from measurement 1.
        mid_j (numpy.ndarray of int|str, optional): GDP model from measurement 2.
        rid_i (numpy.ndarray of int|str, optional): rig id of measurement 1.
        rid_j (numpy.ndarray of int|str, optional): rig id of measurement 2.
        eid_i (numpy.ndarray of int|str, optional): event id of measurement 1.
        eid_j (numpy.ndarray of int|str, optional): event id of measurement 2.

    Warning:
        - If no oids are specified, the function will assume that the data originates
        **from the exact same radiosonde.** Idem for the GDP models, rig id and event id.
        - The profiles are assumed to be synchronized, i.e. if specifying i and j as steps,
        i=j implies that they both have the same altitude.

    Returns:
        numpy.ndarray of float(s): the correlation coefficient(s), in the range [0, 1].

    Note:
        This function returns the pair-wise correlation coefficients,
        **not** the full correlation matrix, i.e::

            shape(corcoef_gdps(i, j, uc_type)) == shape(i) == shape(j)

    The supported uncertainty types are:

    - 'ucr': rig-correlated uncertainty.
             Intended for the so-called "uncorrelated" GRUAN uncertainty.
    - 'ucs': spatial-correlated uncertainty.
             Full correlation between measurements acquired during the same event at the same site,
             irrespective of the time step/altitude, rig, radiosonde model, or serial number.
    - 'uct': temporal-correlated uncertainty.
             Full correlation between measurements acquired at distinct sites during distinct events,
             with distinct radiosondes models and serial numbers.
    - 'ucu': uncorrelated.
             No correlation whatsoever between distinct measurements.

    Todo:
        - Add reference to GRUAN docs & dvas articles in this docstring.

    '''

    # Begin with some safety checks
    for var in [i, j, oid_i, oid_j, oid_i, oid_j, rid_i, rid_j, eid_i, eid_j]:
        if var is None:
            continue
        if not isinstance(var, np.ndarray):
            raise DvasError('Ouch ! I was expecting a numpy.ndarray, not %s' % type(var))
        if np.shape(var) != np.shape(i):
            raise DvasError('Ouch ! All items should have the same shape !')

    # Make sure to return something with the same shape as what came in.
    corcoef = np.zeros_like(i)

    # All variables always correlate fully with themselves
    corcoef[(i == j) * (oid_i == oid_j) * (mid_i == mid_j) *
            (rid_i == rid_j) * (eid_i == eid_j)] = 1.0

    # Now work in the required level of correlation depending on the uncertainty type.
    # TODO: confirm that all of those rules are actually correct !
    if sigma_name == 'ucu':
        # Nothing to add in case of uncorrelated uncertainties.
        pass

    elif sigma_name == 'ucr':
        logger.warning('Rig-correlated uncertainties not yet defined.')

    elif sigma_name == 'ucs':
        # 1) Full spatial-correlation between measurements acquired in the same event
        #    irrespective of the rig number, GDP model or serial number.
        corcoef[(eid_i == eid_j)] = 1.0

    elif sigma_name == 'uct':
        # 1) Full temporal-correlation between measurements acquired in the same event and at the
        #    same site.
        corcoef[(eid_i == eid_j)] = 1.0

        # 2) Full temporal correlation between measurements acquired in distinct events and sites.
        corcoef = np.ones_like(corcoef)

    else:
        raise DvasError("Ouch ! uc_type must be one of ['ucr', 'ucs', 'uct', 'ucu'], not: %s" %
                        (sigma_name))

    return corcoef

@log_func_call(logger)
def weighted_mean(df_chunk, binning=1):
    """ Compute the (respective) weighted mean of the 'tdt', 'val', and 'alt' columns of a
    pd.DataFrame, with weights definied in the 'w_ps' column. Also returns the Jacobian matrix for
    `val` to enable accurate error propagation.

    Note:
        The input format for `df_chunk` is a `pandas.DataFrame` with a very specific structure.
        It requires a single index called `_idx`, with 10 columns per profiles with labels `tdt`,
        `alt`, `val`, 'flg', `ucr`, `ucs`, `uct`, `ucu`, `uc_tot` and `w_ps`. All these must be
        grouped together using pd.MultiIndex where the level 0 corresponds to the profile number
        (e.g. 0,1,2...), and the level 1 is the original column name, i.e.::

                            0                                             ...         1                        0          1
                          alt              tdt         val       ucr ucs  ...       uct ucu    uc_tot       w_ps       w_ps
            _idx                                                          ...
            0      486.726685  0 days 00:00:00  284.784546       NaN NaN  ...  0.100106 NaN  0.211856  55.861518  22.280075
            1      492.425507  0 days 00:00:01  284.695190  0.079443 NaN  ...  0.100106 NaN  0.194927  67.896245  26.318107
            ...

    Args:
        df_chunk (pandas.DataFrame): data conatining the Profiles to merge.
        binning (int, optional): binning size. Defaults to 1 (=no binning).

    Returns:
        (pandas.DataFrame, np.array): weighted mean profile, and associated Jacobian matrix.
        The matrix has a size of m * n, with m = len(df_chunk)/binning, and
        n = len(df_chunk) * n_profile.

    Note:
        The function will ignore NaNs in a given bin, unless *all* the values in the bin are NaNs.
        See fancy_nansum() for details.

    """

    # Begin with some important sanity checks to make sure the DataFrame has the correct format
    for col in [PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME, PRF_REF_FLG_NAME,
                PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME,
                'uc_tot', 'w_ps']:
        if col not in df_chunk.columns.unique(level=1):
            raise DvasError('Ouch ! column "{}" is missing from the DataFrame'.format(col))

    # How many profiles do I want to combine ?
    n_prf = len(df_chunk.columns.unique(level=0))

    # Let's make sure their ID is what I expect them to be.
    if not np.array_equal(df_chunk.columns.unique(level=0), range(n_prf)):
        raise DvasError('Ouch ! Profile values must be grouped usign MultiIndex with ids 0,1, ...')

    # Force the weights to be NaNs if the data is a NaN. Else, the normalization will be off.
    mask = df_chunk.xs(PRF_REF_VAL_NAME, level=1, axis=1).isna()
    df_chunk.loc[:, (slice(None), 'w_ps')] = \
        df_chunk.xs('w_ps', level=1, axis=1).mask(mask, other=np.nan, inplace=False).values

    # Create the structure that will store all the weighted means
    chunk_out = pd.DataFrame()

    # Let's start computing the weighted average. First prepare the weight matrices.
    # 1) Extract the weights
    w_ps = df_chunk.xs('w_ps', level=1, axis=1)

    # 2) Sum them accross profiles (no binning yet)
    w_s = fancy_nansum(w_ps, axis=1)

    # 3) Now deal with the binning if applicable
    if binning > 1:
        w_ms = w_s.groupby(w_s.index//binning).aggregate(fancy_nansum)
    else:
        w_ms = w_s

    # 4) If any of the total weight is 0, replace it with nan's (to avoid runtime Warnings).
    w_ms.mask(w_ms == 0, other=np.nan, inplace=True)

    # Now that we have the weights sorted, we can loop through the variables and computed their
    # weighted means.
    for col in [PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME]:

        # 1) First multiple vals * weights and keep this in a "clean" DataFrame
        wx_ps = df_chunk.xs(col, level=1, axis=1) * w_ps

        # 2) Sum val * weight accross profiles.
        # Note the special treatment of NaNs: ignored, unless that is all I get from all the
        # profiles at a given time step/altitude.
        wx_s = fancy_nansum(wx_ps, axis=1)

        # 3) Then sum these across the time/altitude layers according to the binning
        if binning > 1:
            # Note again the special treatment of NaNs:
            # ignored, unless that is all I have in a given bin.
            wx_ms = wx_s.groupby(wx_s.index//binning).aggregate(fancy_nansum)
        else:
            # If no binning is required, then do nothing and save us some time.
            wx_ms = wx_s

        # Compute the weighted mean
        # To avoid some runtime Warning, we replaced any 0 total weight with nan
        chunk_out[col] = wx_ms / w_ms

    # All done. Let us now compute the associated Jacobian matrix.

    # First, the full matrix of weights
    # Mind the "T" to keep things in order !
    w_mat = np.tile(df_chunk.xs('w_ps', level=1, axis=1).values.T.flatten(), (len(chunk_out), 1))

    # Next we need to know which elements are being combined in each level of the final (binned)
    # profile (the matrix should be mostly filled with 0)

    # --- Kept for legacy purposes ---
    # This is slow
    #in_lvl = np.array([vals.shape[1]*[1 if j//binning == i else 0 for j in range(len(vals))]
    #                   for i in range(len(x_ms))])
    # This is better
    #in_lvl = np.array([vals.shape[1]*([0]*i*binning + [1]*binning + [0]*(len(vals)-1-i)*binning)
    #                   for i in range(len(x_ms))])
    # Apply the selection to the weight matrix. We avoid the multiplication to properly deal with
    # NaNs.
    #w_mat[in_lvl == 0] = 0
    # --------------------------------

    # This is the fastest way to do so I could come up with so far. Re-compute which layer goes
    # where, given the binning.
    rows, cols = np.indices((len(chunk_out), len(df_chunk)*n_prf))
    w_mat[(cols % len(df_chunk))//binning != rows] = 0

    # I also need to assemble a matrix of the total weights for each (final) row.
    wtot_mat = np.tile(w_ms.values, (len(df_chunk)*n_prf, 1)).T

    # I can now assemble the Jacobian for the weighted mean
    jac_mat = w_mat / wtot_mat

    return chunk_out, jac_mat

@log_func_call(logger)
def delta(df_chunk, binning=1):
    """ Compute the delta of the 'tdt', 'val', and 'alt' columns of a pd.DataFrame containing
    exactly 2 Profiles. Also returns the Jacobian matrix for `val` to enable accurate error
    propagation.

    Note:
        The input format for `df_chunk` is a `pandas.DataFrame` with a very specific structure.
        It requires a single index called `_idx`, with 10 columns per profiles with labels `tdt`,
        `alt`, `val`, 'flg', `ucr`, `ucs`, `uct`, `ucu`, `uc_tot` and `w_ps`. All these must be
        grouped together using pd.MultiIndex where the level 0 corresponds to the profile number
        (e.g. 0,1,2...), and the level 1 is the original column name, i.e.::

                            0                                             ...         1                        0          1
                          alt              tdt         val       ucr ucs  ...       uct ucu    uc_tot       w_ps       w_ps
            _idx                                                          ...
            0      486.726685  0 days 00:00:00  284.784546       NaN NaN  ...  0.100106 NaN  0.211856  55.861518  22.280075
            1      492.425507  0 days 00:00:01  284.695190  0.079443 NaN  ...  0.100106 NaN  0.194927  67.896245  26.318107
            ...

    Args:
        df_chunk (pandas.DataFrame): data conatining the Profiles to merge.
        binning (int, optional): binning size. Defaults to 1 (=no binning).

    Returns:
        (pandas.DataFrame, np.array): delta profile (0 - 1), and associated Jacobian matrix.
        The matrix has a size of m * n, with m = len(df_chunk)/binning, and
        n = len(df_chunk) * 2.

    Note:
        The function will ignore NaNs in a given bin, unless *all* the values in the bin are NaNs.
        See fancy_nansum() for details.

    """

    # Begin with some important sanity checks to make sure the DataFrame has the correct format
    for col in [PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME, PRF_REF_FLG_NAME,
                PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME,
                'uc_tot']:
        if col not in df_chunk.columns.unique(level=1):
            raise DvasError('Ouch ! column "{}" is missing from the DataFrame'.format(col))

    # How many profiles do I want to combine ?
    n_prf = len(df_chunk.columns.unique(level=0))

    if n_prf != 2:
        raise DvasError("Ouch ! I can only make the difference between 2 profiles, " +
                        "but you gave me {}.".format(n_prf))

    # TODO: everything after this needs fixing. Question: what do I do with the alt and tdt ?


    # Compute the difference between the two profiles (full resolution)
    delta_pqs = vals[0] - vals[1]

    if binning > 1:
        # If required, bin the array as required.
        # Note the special treatment of NaNs: ignored, unless that is all I have in a given bin.
        delta_pqm = delta_pqs.groupby(delta_pqs.index//binning).aggregate(fancy_nansum)

        # Keep track of how many valid rows I am summing in each bin.
        valid_rows = pd.Series(np.ones_like(delta_pqs)).mask(delta_pqs.isna(), 0)
        valid_rows = valid_rows.groupby(delta_pqs.index//binning).sum()

        # Build the mean by normalizing the sum by the number of time/altitude steps combined
        x_ms = delta_pqm / valid_rows

    else:
        # If no binning is required, I can save *a lot* of time
        x_ms = delta_pqs

    # Let us now compute the non-zero Jacobian matrix elements.
    # See the comments in the weighted_mean() function regardin the motivation for not computing or
    # returning the full Jacobian matrix (which is pretty much filled with 0's.)

    jac_elmts = [len(x_ms) * ([1/binning for i in range(binning)] +
                              [-1/binning for i in range(binning)])]

    return x_ms, jac_elmts

def process_chunk(df_chunk, binning=1, method='weighted mean'):
    """ Process a DataFrame chunk and propagate the errors.

    Note:
        The input format for `df_chunk` is a `pandas.DataFrame` with a very specific structure.
        It requires a single index called `_idx`, with 10 columns per profiles with labels `tdt`,
        `alt`, `val`, 'flg', `ucr`, `ucs`, `uct`, `ucu`, `uc_tot` and `w_ps`. All these must be
        grouped together using pd.MultiIndex where the level 0 corresponds to the profile number
        (e.g. 0,1,2...), and the level 1 is the original column name, i.e.::

                            0                                             ...         1                        0          1
                          alt              tdt         val       ucr ucs  ...       uct ucu    uc_tot       w_ps       w_ps
            _idx                                                          ...
            0      486.726685  0 days 00:00:00  284.784546       NaN NaN  ...  0.100106 NaN  0.211856  55.861518  22.280075
            1      492.425507  0 days 00:00:01  284.695190  0.079443 NaN  ...  0.100106 NaN  0.194927  67.896245  26.318107
            ...

    Args:
        df_chunk (pandas.DataFrame): data conatining the Profiles to merge.
        binning (int, optional): binning size. Defaults to 1 (=no binning).
        method (str, optional): the processing method. Can be one of
            ['mean', 'weighted mean', delta']. Defaults to 'weighted mean'.

    Returns:
        pandas.DataFrame: the processing outcome, including all the errors.

    Note:
        The function will ignore NaNs in a given bin, unless *all* the values in the bin are NaNs.
        See fancy_nansum() for details.
    """

    # Begin with some sanity checks
    if method not in ['mean', 'weighted mean', 'delta']:
        raise DvasError('Ouch! method {} unknown.'.format(method))

    # Check I have all the required columns
    for col in [PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME, PRF_REF_FLG_NAME,
                PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME,
                'uc_tot']:
        if col not in df_chunk.columns.unique(level=1):
            raise DvasError('Ouch ! column "{}" is missing from the DataFrame'.format(col))

    # How many profiles do I want to combine ?
    n_prf = len(df_chunk.columns.unique(level=0))

    # Compute the weights for each point, if applicable
    # First I need to add the new columns (one for each Profile).
    df_chunk = df_chunk.stack(level=0)
    df_chunk.loc[:, 'w_ps'] = np.nan
    df_chunk = df_chunk.unstack().swaplevel(0, 1, axis=1).sort_index(axis=1)

    # Then I can fill it with the appropriate values
    if method == 'weighted mean':
        df_chunk.loc[:, (slice(None), 'w_ps')] = \
            1/df_chunk.loc[:, (slice(None), 'uc_tot')].values**2
    else:
        df_chunk.loc[:, (slice(None), 'w_ps')] = 1.

    # Combine the Profiles, and compute the Jacobian matrix as well
    if method in ['mean', 'weighted mean']:
        x_ms, G_mat = weighted_mean(df_chunk, binning=binning)
    elif method == 'delta':
        x_ms, G_mat = delta(df_chunk, binning=binning)

    # Let's get started with the computation of the errors.
    # Let us now assemble the U matrices, filling all the cross-correlations for the different
    # types of uncertainties
    for sigma_name in [PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME]:
        U_mat = corcoefs(np.tile(df_chunk.index.values, (n_prf*len(df_chunk), n_prf)), # i
                         np.tile(df_chunk.index.values, (n_prf*len(df_chunk), n_prf)).T, # j
                         sigma_name,
                         oid_i=np.tile(df_chunk.xs('oid', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)),
                         oid_j=np.tile(df_chunk.xs('oid', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)).T,
                         mid_i=np.tile(df_chunk.xs('mdl_id', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)),
                         mid_j=np.tile(df_chunk.xs('mdl_id', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)).T,
                         rid_i=np.tile(df_chunk.xs('rig_id', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)),
                         rid_j=np.tile(df_chunk.xs('rig_id', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)).T,
                         eid_i=np.tile(df_chunk.xs('evt_id', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)),
                         eid_j=np.tile(df_chunk.xs('evt_id', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)).T,
                        )

        # Implement the multiplication. Mind the structure of these arrays to get the correct
        # mix of Hadamard and dot products where I need them !
        raveled_sigmas = np.array([df_chunk.xs(sigma_name, level=1, axis=1).T.values.ravel()])
        U_mat = np.multiply(U_mat, raveled_sigmas.T @ raveled_sigmas)

        # Let's compute the full covariance matrix for the merged profile (for the specific
        # error type).
        # This is a square matrix, with the off-axis elements containing the covarience terms
        # for the merged profile.
        V_mat = np.where(np.isnan(G_mat), 0, G_mat) @ \
                np.where(np.isnan(U_mat), 0, U_mat) @ \
                np.where(np.isnan(G_mat.T), 0, G_mat.T)

        # As a sanity check let's make sure all the off-diagonal terms are exactly 0.
        # This should be the case since a specific (original) layer can only be used once
        # in the combined profile.
        if not np.array_equal(V_mat[(V_mat != 0)|(np.isnan(V_mat))],
                              V_mat.diagonal(), equal_nan=True):
            logger.warning("Non-0 off-diagonal elements of CWS correlation matrix [%s].",
                           sigma_name)

        # TODO: Deal with NaN's properly

        # I can finally use matrix multiplication (using @) to save me a lot of convoluted sums ...
        # If I only have nan's then that's my result. If I only have partial nan's, then make sure I
        # still get a number out.
        #if np.all(np.isnan(jac_elmts)):
        #    comb_ucrs[k_ind] = np.nan
        #    comb_ucss[k_ind] = np.nan
        #    comb_ucts[k_ind] = np.nan
        #    comb_ucus[k_ind] = np.nan

        #else:
        # Replace all the nan's with zeros so they do not intervene in the sums
        #if np.all(np.isnan(u_mats[0])):
        #    comb_ucrs[k_ind] = np.nan
        #else:
        #    comb_ucrs[k_ind] = np.where(np.isnan(jac_elmts[k_ind]), 0, jac_elmts[k_ind]) @ \
        #                          np.where(np.isnan(u_mats[0]), 0, u_mats[0]) @ \
        #                          np.where(np.isnan(jac_elmts[k_ind].T), 0, jac_elmts[k_ind].T)
        #    comb_ucrs[k_ind] = np.sqrt(comb_ucrs[k_ind])

        # Assign the values to the combined df_chunk
        x_ms.loc[:, sigma_name] = np.sqrt(V_mat.diagonal())

    # All done
    return x_ms
