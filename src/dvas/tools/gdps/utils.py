# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains GRUAN-related utilities.

"""

# WARNING: this package should NOT import anything from dvas.data.
# The routines included must be designed in a general manner, such that they can also be
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
    pd.DataFrame, with weights defined in the 'w_ps' column. Also returns the Jacobian matrix for
    `val` to enable accurate error propagation.

    Note:
        The input format for `df_chunk` is a `pandas.DataFrame` with a very specific structure.
        It requires a single index called `_idx`, with 5 columns per profiles with labels `tdt`,
        `alt`, `val`, 'flg', and `w_ps`. All these must be grouped together using pd.MultiIndex
        where the level 0 corresponds to the profile number (e.g. 0, 1, 2...), and the level 1 is
        the original column name, i.e.::

                       0                                        1
                     alt              tdt    val  flg  w_ps   alt  ... w_ps
            _idx
            0      486.7  0 days 00:00:00  284.7    0  55.8  485.9 ... 22.4
            1      492.4  0 days 00:00:01  284.6    1  67.5  493.4 ... 26.3
            ...

    Args:
        df_chunk (pandas.DataFrame): data conatining the Profiles to merge.
        binning (int, optional): binning size. Defaults to 1 (=no binning).

    Returns:
        (pandas.DataFrame, np.ma.masked_array): weighted mean profile, and associated Jacobian
        matrix. The matrix has a size of m * n, with m = len(df_chunk)/binning, and
        n = len(df_chunk) * n_profile.

    Note:
        The function will ignore NaNs in a given bin, unless *all* the values in the bin are NaNs.
        See fancy_nansum() for details.

    """

    # Begin with some important sanity checks to make sure the DataFrame has the correct format
    for col in [PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME, PRF_REF_FLG_NAME, 'w_ps']:
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
    # Anything not in a bin has a NaN weight. This is important to distinguish from an element with
    # zero weight but that get included in the bin.
    w_mat[(cols % len(df_chunk))//binning != rows] = np.nan

    # I also need to assemble a matrix of the total weights for each (final) row.
    wtot_mat = np.tile(w_ms.values, (len(df_chunk)*n_prf, 1)).T

    # I can now assemble the Jacobian for the weighted mean. Turn this into a masked array.
    jac_mat = np.ma.masked_array(w_mat / wtot_mat,
                                 mask=(np.isnan(wtot_mat) | np.isnan(w_mat)),
                                 fill_value=np.nan)

    return chunk_out, jac_mat

@log_func_call(logger)
def delta(df_chunk, binning=1):
    """ Compute the delta of the 'tdt', 'val', and 'alt' columns of a pd.DataFrame containing
    exactly 2 Profiles. Also returns the Jacobian matrix for `val` to enable accurate error
    propagation.

    Note:
        The input format for `df_chunk` is a `pandas.DataFrame` with a very specific structure.
        It requires a single index called `_idx`, with 4 columns per profiles with labels `tdt`,
        `alt`, `val`, and 'flg'. All these must be grouped together using pd.MultiIndex where the
        level 0 corresponds to the profile number (i.e. 0 or 1), and the level 1 is the original
        column name, i.e.::

                       0                               1
                     alt              tdt    val flg alt    ...  flg
            _idx
            0      486.7  0 days 00:00:00  284.7   0 486.5  ...    0
            1      492.4  0 days 00:00:01  284.6   0 491.9  ...    1
            ...

    Args:
        df_chunk (pandas.DataFrame): data containing the Profiles to merge.
        binning (int, optional): binning size. Defaults to 1 (=no binning).

    Returns:
        (pandas.DataFrame, np.ma.masked_array): delta profile (0 - 1), and associated Jacobian
        matrix. The matrix has a size of m * n, with m = len(df_chunk)/binning, and
        n = len(df_chunk) * 2.

    Note:
        The function will ignore NaNs in a given bin, unless *all* the values in the bin are NaNs.
        See fancy_nansum() for details.

    """

    # Begin with some important sanity checks to make sure the DataFrame has the correct format
    for col in [PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME, PRF_REF_FLG_NAME]:
        if col not in df_chunk.columns.unique(level=1):
            raise DvasError('Ouch ! column "{}" is missing from the DataFrame'.format(col))

    # How many profiles do I want to combine ?
    n_prf = len(df_chunk.columns.unique(level=0))

    if n_prf != 2:
        raise DvasError("Ouch ! I can only make the difference between 2 profiles, " +
                        "but you gave me {}.".format(n_prf))

    # Create the structure that will store all the weighted means
    chunk_out = pd.DataFrame()

    # Let's loop through the variables and compute their deltas.
    for col in [PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME]:

        # 1) Build the delta at high resolution
        delta_pqs = df_chunk.xs(col, level=1, axis=1).diff(axis=1).loc[:, 1]

        if binning > 1:
            # If required, bin the array.
            # Note the special treatment of NaNs: ignored, unless that is all I have in a given bin.
            delta_pqm = delta_pqs.groupby(delta_pqs.index//binning).aggregate(fancy_nansum)

            # Keep track of how many valid rows I am summing in each bin.
            # It would seem logical to store these as int. However, I will soon divide by those
            # numbers, and will need NaN's to avoid RunTime Warnings. 'int' does not support these,
            # and 'Int64' cannot be used to divide TiemDeltas ... so float it is ... sigh.
            # WARNING: here, i look at the number of valid rows **for that specific column**.
            # This implies that, potentially, the delta action may combine different rows for
            # different columns.
            # TODO: fix this ... by using flags to select valid rows ?
            valid_rows = pd.Series(np.ones(len(delta_pqs)),
                                   dtype='float').mask(delta_pqs.isna().values, 0)
            valid_rows = valid_rows.groupby(delta_pqs.index//binning).sum()

            # Build the mean by normalizing the sum by the number of time/altitude steps combined
            x_ms = delta_pqm / valid_rows.mask(valid_rows == 0, np.nan)

        else:
            # If no binning is required, I can save *a lot* of time
            x_ms = delta_pqs
            valid_rows = pd.Series(np.ones(len(x_ms)), dtype='float')

        # Assign the delta
        chunk_out[col] = x_ms

        # All done. Let us now compute the associated Jacobian matrix if we are dealing with 'val'.
        if col != PRF_REF_VAL_NAME:
            continue

        # How big is my Jacobian ? (Make it a masked_array)
        jac_mat = np.ma.masked_array(np.ones((len(chunk_out), len(df_chunk)*n_prf)),
                                     mask=False,
                                     fill_value=np.nan)

        # We're doing 0 - 1, so let's already set the second half of the matrix accordingly.
        jac_mat[:, len(df_chunk):] = -1

        # Next we need to know which elements are being combined in each level of the final
        # (binned) profile (the matrix should be mostly filled with 0).

        # This is the fastest way to do so I could come up with so far. Re-compute which layer
        # goes where, given the binning.
        rows, cols = np.indices((len(chunk_out), len(df_chunk)*n_prf))
        # Anything not in a bin has a NaN weight. This is important to distinguish from an element
        # with zero weight but that gets included in the bin.
        jac_mat[(cols % len(df_chunk))//binning != rows] = np.ma.masked

        # Next, I need to mask any level that is completely a NaN, because it does not get used in
        # the computation. This is important, when binning, if e.g. 1/4 of the bin is Nan,
        # To have a correct error propagation.
        points_to_forget = delta_pqs.append(delta_pqs).isna().values
        jac_mat[:, points_to_forget] = np.ma.masked

        # Finally, let us not forget that we may have averaged the delta over the bin ...
        # Normalize by the number of valid levels if warranted.
        # Here, I need to go back from Int64 to float, because I need to turn <NA> into usual NaNs
        # if I want to divide a float with them. ... sigh ...
        jac_mat /= np.array([valid_rows.mask(valid_rows == 0, np.nan).values]).T

        # Adjust the mask accordingly.
        jac_mat[valid_rows == 0, :] = np.ma.masked

    return chunk_out, jac_mat

def process_chunk(df_chunk, binning=1, method='weighted mean'):
    """ Process a DataFrame chunk and propagate the errors.

    Note:
        The input format for `df_chunk` is a `pandas.DataFrame` with a very specific structure.
        It requires a single index called `_idx`, with 13 columns per profiles with labels `tdt`,
        `alt`, `val`, 'flg', `ucr`, `ucs`, `uct`, `ucu`, `uc_tot`, `oid`, `mid`, `eid`, and `rid`.
        All these must be grouped together using pd.MultiIndex where the level 0 corresponds
        to the profile number (e.g. 0,1,2...), and the level 1 is the original column name, i.e.::

                       0                                                1
                     alt              tdt    val   ucr ucs  ...   rid alt ...
            _idx
            0      486.7  0 days 00:00:00  284.7   NaN NaN  ...     1 485.8
            1      492.4  0 days 00:00:01  284.6  0.07 NaN  ...     1 493.4
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
                'uc_tot', 'oid', 'mid', 'eid', 'rid']:
        if col not in df_chunk.columns.unique(level=1):
            raise DvasError('Ouch ! column "{}" is missing from the DataFrame'.format(col))

    # How many profiles do I want to combine ?
    n_prf = len(df_chunk.columns.unique(level=0))

    # Data consistency check & cleanup: if val is NaN, all the errors should be NaNs
    # (and vice-versa).
    for prf_ind in range(n_prf):
        if all(df_chunk.loc[:, (prf_ind, 'uc_tot')].isna() ==
               df_chunk.loc[:, (prf_ind, 'val')].isna()):
            continue

        # If I reach this point, then the data is not making sense. Warn the user and clean it up.
        logger.warning("GDP Profile %i: NaN mismatch for 'val' and 'uc_tot'", prf_ind)
        df_chunk.loc[df_chunk.loc[:, (prf_ind, 'uc_tot')].isna().values, (prf_ind, 'val')] = np.nan
        for col in [PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME,
                    'uc_tot']:
            df_chunk.loc[df_chunk.loc[:, (prf_ind, 'val')].isna().values,
                         (prf_ind, col)] = np.nan

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
        # Let us not forget to mask anything that has a NaN value or total error.
        df_chunk.loc[:, (slice(None), 'w_ps')] = df_chunk.loc[:, (slice(None), 'w_ps')].mask(
            df_chunk.loc[:, (slice(None), 'val')].isna().values, np.nan)
        df_chunk.loc[:, (slice(None), 'w_ps')] = df_chunk.loc[:, (slice(None), 'w_ps')].mask(
            df_chunk.loc[:, (slice(None), 'uc_tot')].isna().values, np.nan)

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
                         mid_i=np.tile(df_chunk.xs('mid', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)),
                         mid_j=np.tile(df_chunk.xs('mid', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)).T,
                         rid_i=np.tile(df_chunk.xs('rid', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)),
                         rid_j=np.tile(df_chunk.xs('rid', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)).T,
                         eid_i=np.tile(df_chunk.xs('eid', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)),
                         eid_j=np.tile(df_chunk.xs('eid', level=1, axis=1).T.values.ravel(),
                                       (n_prf*len(df_chunk), 1)).T,
                        )

        # Implement the multiplication. First get the uncertainties ...
        raveled_sigmas = np.array([df_chunk.xs(sigma_name, level=1, axis=1).T.values.ravel()])
        # ... turn them into a masked array ...
        raveled_sigmas = np.ma.masked_invalid(raveled_sigmas)
        # ... and combine them with the correlation coefficients. Mind the mix of Hadamard and dot
        # products to get the correct mix !
        U_mat = np.multiply(U_mat, np.ma.dot(raveled_sigmas.T, raveled_sigmas))

        # Let's compute the full covariance matrix for the merged profile (for the specific
        # error type).
        # This is a square matrix, with the off-axis elements containing the covarience terms
        # for the merged profile. All these matrices are masked arrays, such that bad values will be
        # correctly ignored .... unless that is all we have for a given bin.
        V_mat = np.ma.dot(G_mat, np.ma.dot(U_mat, G_mat.T))

        # As a sanity check let's make sure all the off-diagonal terms are exactly 0.
        # This should be the case since a specific (original) layer can only be used once
        # in the combined profile.
        rows, cols = np.indices((len(x_ms), len(x_ms)))
        off_diag_elmts = V_mat[rows != cols]
        if any(off_diag_elmts[~off_diag_elmts.mask] != 0):
            logger.warning("Non-0 off-diagonal elements of CWS correlation matrix [%s].",
                           sigma_name)

        # TODO: what could we do with the off-diagonal elements of V_mat ? Nothing for now.

        # Assign the propagated uncertainty values to the combined df_chunk, taking care of
        # replacing any masked element with NaNs.
        x_ms.loc[:, sigma_name] = np.sqrt(V_mat.diagonal().filled(np.nan))

    # All done
    return x_ms
