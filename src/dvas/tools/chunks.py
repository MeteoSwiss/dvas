# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains GRUAN-related utilities.

"""

# Import from Python
import logging
import numpy as np
import pandas as pd

# Import from current package
# from ..logger import log_func_call
from ..errors import DvasError
from ..hardcoded import PRF_TDT, PRF_ALT, PRF_VAL, PRF_FLG, PRF_UCS, PRF_UCT, PRF_UCU
from .tools import fancy_nansum, fancy_bitwise_or, wrap_angle
from .gdps.correlations import corr_coeff_matrix

# Setup local logger
logger = logging.getLogger(__name__)


#@log_func_call(logger)
def merge_bin(wx_ps, binning):
    """ Small utility function to sum individual profiles into bins.

    Args:
        wx_ps (pd.DataFrame): the DataFrame to bin, with columns representing distinct profiles.
        binning (int): the vertical binning.

    Returns:
        pd.DataFrame: the binned profile.

    """

    # First, sum accross profiles
    # Note the special treatment of NaNs: ignored, unless that is all I get from all the
    # profiles at a given time step/altitude.
    wx_s = fancy_nansum(wx_ps, axis=1)

    # Then sum these across the time/altitude layers according to the binning
    if binning > 1:
        # Note again the special treatment of NaNs:
        # ignored, unless that is all I have in a given bin.
        return wx_s.groupby(wx_s.index//binning).aggregate(fancy_nansum)

    # If no binning is required, then do nothing and save us some time.
    return wx_s


#@log_func_call(logger)
def weighted_mean(df_chunk, binning=1, mode='arithmetic'):
    """ Compute the (respective) weighted mean of the 'tdt', 'val', and 'alt' columns of
    a pd.DataFrame, with weights defined in the 'w_ps' column. Also returns the Jacobian matrix for
    `val` to enable accurate error propagation.

    Args:
        df_chunk (pandas.DataFrame): data containing the Profiles to merge.
        binning (int, optional): binning size. Defaults to 1 (=no binning).
        mode (str, optional): whether to compute an arithmetic or circular mean for 'val'.
            An arithmetic mean is always computed for 'tdt' and 'alt'.

    Returns:
        (pandas.DataFrame, np.ma.masked_array): weighted mean profile, and associated Jacobian
        matrix. The matrix has a size of m * n, with m = len(df_chunk)/binning, and
        n = len(df_chunk) * n_profile.

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

    Note:
        The function will ignore NaNs in a given bin, unless *all* the values in the bin are NaNs.
        See fancy_nansum() for details.

    """

    # Begin with some important sanity checks to make sure the DataFrame has the correct format
    for col in [PRF_TDT, PRF_ALT, PRF_VAL, PRF_FLG, 'w_ps']:
        if col not in df_chunk.columns.unique(level=1):
            raise DvasError(f'Column "{col}" is missing from the DataFrame')

    # How many profiles do I want to combine ?
    n_prf = len(df_chunk.columns.unique(level=0))

    # Let's make sure their ID is what I expect them to be.
    if not np.array_equal(df_chunk.columns.unique(level=0), range(n_prf)):
        raise DvasError('Profile values must be grouped using MultiIndex with ids 0,1, ...')

    # Force the weights to be NaNs if the data is a NaN. Else, the normalization will be off.
    mask = df_chunk.xs(PRF_VAL, level=1, axis=1).isna()
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
    for col in [PRF_TDT, PRF_ALT, PRF_VAL]:

        if col in [PRF_VAL] and mode == 'circular':
            # Compute the circular mean
            # 1) Compute u and v
            u_ps = df_chunk.xs(col, level=1, axis=1).apply(np.deg2rad).apply(np.sin) * w_ps
            v_ps = df_chunk.xs(col, level=1, axis=1).apply(np.deg2rad).apply(np.cos) * w_ps

            # 2) Sum u and v bins.
            u_ms = merge_bin(u_ps, binning)
            v_ms = merge_bin(v_ps, binning)

            # 3) Compute the weighted circular mean - not forgetting to bring this back in the
            # range [0, 360[.
            chunk_out[col] = np.arctan2(u_ms, v_ms).apply(np.rad2deg) % 360

        else:
            # Compute the arithmetic mean

            # 1) First multiple vals * weights and keep this in a "clean" DataFrame
            wx_ps = df_chunk.xs(col, level=1, axis=1) * w_ps

            # 2) Sum the bins
            wx_ms = merge_bin(wx_ps, binning)

            # 3) Compute the weighted arithmetic mean
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
    # in_lvl = np.array([vals.shape[1]*[1 if j//binning == i else 0 for j in range(len(vals))]
    #                   for i in range(len(x_ms))])
    # This is better
    # in_lvl = np.array([vals.shape[1]*([0]*i*binning + [1]*binning + [0]*(len(vals)-1-i)*binning)
    #                   for i in range(len(x_ms))])
    # Apply the selection to the weight matrix. We avoid the multiplication to properly deal with
    # NaNs.
    # w_mat[in_lvl == 0] = 0
    # --------------------------------

    # This is the fastest way to do so I could come up with so far. Re-compute which layer goes
    # where, given the binning.
    rows, cols = np.indices((len(chunk_out), len(df_chunk)*n_prf))
    # Anything not in a bin has a NaN weight. This is important to distinguish from an element with
    # zero weight but that get included in the bin.
    w_mat[(cols % len(df_chunk))//binning != rows] = np.nan

    if mode == 'circular':

        # Assemble the full matrix of angles ...
        x_mat = np.tile(df_chunk.xs('val', level=1, axis=1).values.T.flatten(), (len(chunk_out), 1))
        # ... in radiansm, ready to be sin-ed or cos-ed
        x_mat = np.deg2rad(x_mat)

        # Compute some important factors (see the LaTeX doc of Hell for details)
        v_u2v2 = np.tile(v_ms/(u_ms**2+v_ms**2).values, (len(df_chunk)*n_prf, 1)).T
        u_v = np.tile(u_ms/v_ms, (len(df_chunk)*n_prf, 1)).T

        # Here, we shall mask any infity that may occur ...
        # typically if I get two angles 180deg appart
        u_v[np.isinf(u_v)] = np.nan

        # Ready to assemble the Jacobian matrix. We make turn it into a masked array.
        jac_mat = np.ma.masked_array(v_u2v2 * (w_mat*np.cos(x_mat) + u_v*w_mat*np.sin(x_mat)),
                                     mask=(np.isnan(v_u2v2) | np.isnan(w_mat) | np.isnan(x_mat) |
                                           np.isnan(u_v)),
                                     fill_value=np.nan)

        # In very bad cases (e.g. 0 & 180), the Jacobian values can be NaN. In those cases, let's
        # also set the chunk value to NaN.
        chunk_out[jac_mat.sum(axis=1).mask] = np.nan

    else:
        # Get the Jacobian for the arithmetic mean
        # I need to assemble a matrix of the total weights for each (final) row.
        wtot_mat = np.tile(w_ms.values, (len(df_chunk)*n_prf, 1)).T

        # I can now assemble the Jacobian for the weighted mean. Turn this into a masked array.
        jac_mat = np.ma.masked_array(w_mat / wtot_mat,
                                     mask=(np.isnan(wtot_mat) | np.isnan(w_mat)),
                                     fill_value=np.nan)

    # Before we end, let us compute the flags. We apply a general bitwise OR to them, such that
    # they do not cancel each other or disappear: they get propagated all the way
    # First, we assemble them at high resolution. Note here that the flag for any weight that is
    # NaN or 0 is set to 0, so they do not get carried over.
    flgs = pd.DataFrame(fancy_bitwise_or(
        df_chunk.loc[:, (slice(None), PRF_FLG)].mask((w_ps.isna() | (w_ps == 0)).values, other=0),
        axis=1))

    # Then, only if warranted, apply the binning too
    if binning > 1:
        flgs = flgs.groupby(flgs.index//binning).aggregate(fancy_bitwise_or)
    chunk_out[PRF_FLG] = flgs.values

    return chunk_out, jac_mat


#@log_func_call(logger)
def delta(df_chunk, binning=1, mode='arithmetic'):
    """ Compute the delta of the 'tdt', 'val', and 'alt' columns of a pd.DataFrame containing
    exactly 2 Profiles. Also returns the Jacobian matrix for `val` to enable accurate error
    propagation.

    Args:
        df_chunk (pandas.DataFrame): data containing the Profiles to merge.
        binning (int, optional): binning size. Defaults to 1 (=no binning).
        mode (str, optional): whether to compute an arithmetic or circular delta for 'val'.
            An arithmetic delta is always computed for 'tdt' and 'alt'. A circular delta will wrap
            results between [-180;180[.

    Returns:
        (pandas.DataFrame, np.ma.masked_array): delta profile (1 - 0), and associated Jacobian
        matrix. The matrix has a size of m * n, with m = len(df_chunk)/binning, and
        n = len(df_chunk) * 2.

    Note:
        The delta is computed as Profile2-Profile1, via a diff() function.

    Note:
        The function will ignore NaNs in a given bin, unless *all* the values in the bin are NaNs.
        See fancy_nansum() for details.

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

    """

    # Begin with some important sanity checks to make sure the DataFrame has the correct format
    for col in [PRF_TDT, PRF_ALT, PRF_VAL, PRF_FLG]:
        if col not in df_chunk.columns.unique(level=1):
            raise DvasError(f'Column "{col}" is missing from the DataFrame')

    # How many profiles do I want to combine ?
    n_prf = len(df_chunk.columns.unique(level=0))

    if n_prf != 2:
        raise DvasError("I can only make the difference between 2 profiles, " +
                        f"but you gave me {n_prf}.")

    # Create the structure that will store all the weighted means
    chunk_out = pd.DataFrame()

    # Let's loop through the variables and compute their deltas.
    for col in [PRF_TDT, PRF_ALT, PRF_VAL]:

        # 1) Build the delta at high resolution
        delta_pqs = df_chunk.xs(col, level=1, axis=1).diff(axis=1).loc[:, 1]

        if binning > 1:
            # If required, bin the array.
            # Note the special treatment of NaNs: ignored, unless that is all I have in a given bin.
            delta_pqm = delta_pqs.groupby(delta_pqs.index//binning).aggregate(fancy_nansum)

            # Keep track of how many valid rows I am summing in each bin.
            # It would seem logical to store these as int. However, I will soon divide by those
            # numbers, and will need NaN's to avoid RunTime Warnings. 'int' does not support these,
            # and 'Int64' cannot be used to divide TimeDeltas ... so float it is ... sigh.
            # WARNING: here, I look at the number of valid rows **for that specific column**.
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
        if col != PRF_VAL:
            continue

        # Wrap angle in a suitable range, if warranted.
        # Since this simply adds/removes a constant, it does not change the Jacobian
        if mode == 'circular':
            chunk_out[col] = chunk_out[col].apply(wrap_angle)

        # How big is my Jacobian ? (Make it a masked_array)
        jac_mat = np.ma.masked_array(np.ones((len(chunk_out), len(df_chunk)*n_prf)),
                                     mask=False,
                                     fill_value=np.nan)

        # We're doing 1 - 0, so let's already set the second half of the matrix accordingly.
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
        points_to_forget = pd.concat([delta_pqs]*2).isna().values
        jac_mat[:, points_to_forget] = np.ma.masked

        # Finally, let us not forget that we may have averaged the delta over the bin ...
        # Normalize by the number of valid levels if warranted.
        # Here, I need to go back from Int64 to float, because I need to turn <NA> into usual NaNs
        # if I want to divide a float with them. ... sigh ...
        jac_mat /= np.array([valid_rows.mask(valid_rows == 0, np.nan).values]).T

        # Adjust the mask accordingly.
        jac_mat[valid_rows == 0, :] = np.ma.masked

    # Before we end, let us compute the flags. We apply a general bitwise OR to them, such that
    # they do not cancel each other or disappear: they get propagated all the way
    # First, we assemble them at high resolution. Note that we here mask any flags that belongs to
    # a NaN value, because this is not actually used in the delta.
    flgs = pd.DataFrame(fancy_bitwise_or(
        df_chunk.loc[:, (slice(None), PRF_FLG)].mask(df_chunk.loc[:, (slice(None),
                                                                      PRF_VAL)].isna().values,
                                                     other=0), axis=1))

    # Then, only if warranted, apply the binning too
    if binning > 1:
        flgs = flgs.groupby(flgs.index//binning).aggregate(fancy_bitwise_or)
    chunk_out[PRF_FLG] = flgs.values

    return chunk_out, jac_mat


def biglambda(df_chunk):
    """ Compute the Lambda value (which is the RMS) of the 'val' column of a pd.DataFrame containing
    a series of measurements, possibly from distinct profiles and out-of-order.
    Also returns the Jacobian matrix for `val` to enable accurate error propagation.

    Args:
        df_chunk (pandas.DataFrame): data to process.

    Returns:
        (pandas.DataFrame, np.ma.masked_array): lambda value, and associated Jacobian
        matrix. The matrix has a size of 1 * n, with n = len(df_chunk).

    Note:
        The input format for `df_chunk` is a `pandas.DataFrame` with a very specific structure.
        It requires a single index called `_idx`, with 4 columns with labels `tdt`,
        `alt`, `val`, and 'flg'. All these must be grouped together using pd.MultiIndex where the
        level 0 corresponds to the profile number which must be 0, and the level 1 is the original
        column name, i.e.::

                       0
                     alt  val flg
            _idx
            0      486.7  284.7   0
            1      492.4  284.6   0
            ...

    """

    # Begin with some important sanity checks to make sure the DataFrame has the correct format
    for col in [PRF_ALT, PRF_VAL, PRF_FLG]:
        if col not in df_chunk.columns.unique(level=1):
            raise DvasError(f'Column "{col}" is missing from the DataFrame')

    # How many profiles do I want to combine ?
    n_prf = len(df_chunk.columns.unique(level=0))

    if n_prf != 1:
        raise DvasError("I expected all the items in a single pseudo-profile, " +
                        f"but you gave me {n_prf}.")

    # Create the structure that will store the Lambda value
    chunk_out = pd.DataFrame()

    # If I was given no data, deal with it
    if len(df_chunk) == 0:
        chunk_out.loc[0, PRF_VAL] = np.nan
        chunk_out.loc[0, 'mean'] = np.nan
        chunk_out.loc[0, 'std'] = np.nan
        chunk_out.loc[0, 'n_pts'] = 0
        chunk_out.loc[0, 'n_prfs'] = 0

        chunk_out['n_pts'] = chunk_out.n_pts.astype(int)
        chunk_out['n_prfs'] = chunk_out.n_prfs.astype(int)

        return chunk_out, None

    # Let's loop through the variables and compute their biglambda value.
    for col in [PRF_VAL]:

        # Compute the RMSE, bias and std. Mind the ddof of pandas std !!!
        chunk_out.loc[0, PRF_VAL] = (df_chunk[(0, PRF_VAL)]**2).mean()**0.5
        chunk_out.loc[0, 'mean'] = (df_chunk[(0, PRF_VAL)]).mean()
        chunk_out.loc[0, 'std'] = (df_chunk[(0, PRF_VAL)]).std(ddof=0)
        # Quick sanity check
        sanity = (chunk_out.loc[:, 'mean']**2+chunk_out.loc[:, 'std']**2)
        sanity = sanity.pow(0.5).round(10).equals(chunk_out.loc[:, PRF_VAL].round(10))
        if not sanity:
            raise DvasError('RMSE vs MEAN + STD mismatch.')
        # Also keep track of the amount of points/profiles used to derive those values
        # Make sure to NOT count any invalid value (e.g. where the CWS of manufacturer is a NaN)
        # as this would mess up the Jacobian matrix (since a 1/J goes in there).
        is_usable = ~df_chunk[(0, 'val')].isna()
        chunk_out.loc[0, 'n_pts'] = len(df_chunk[is_usable])
        chunk_out.loc[0, 'n_prfs'] = \
            len(np.unique(df_chunk[is_usable].loc[:, (0, 'profile_index')].values))

        # All done. Let us now compute the associated Jacobian matrix if we are dealing with 'val'.
        if col != PRF_VAL:
            continue

        # How big is my Jacobian ? (Make it a masked_array)
        jac_mat = np.ma.masked_array(np.ones((len(chunk_out), len(df_chunk))),
                                     mask=False,
                                     fill_value=np.nan)

        # Let's fill it according to the statistical guide
        jac_mat[0, :] = df_chunk.loc[:, (0, PRF_VAL)].values
        jac_mat /= chunk_out.loc[0, 'n_pts']
        jac_mat /= chunk_out.loc[0, PRF_VAL]

        # Let's not forget to hide any bad element
        jac_mat = np.ma.masked_invalid(jac_mat)

    # Before we end, let us compute the flags. We apply a general bitwise OR to them, such that
    # they do not cancel each other or disappear: they get propagated all the way.
    chunk_out[PRF_FLG] = fancy_bitwise_or(df_chunk.loc[:, (slice(None), PRF_FLG)], axis=None)

    chunk_out[PRF_FLG] = chunk_out.loc[:, PRF_FLG].astype(int)
    chunk_out['n_pts'] = chunk_out.n_pts.astype(int)
    chunk_out['n_prfs'] = chunk_out.n_prfs.astype(int)

    return chunk_out, jac_mat


def process_chunk(df_chunk, binning=1, method='weighted arithmetic mean',
                  return_V_mats=True, cov_mat_max_side=10000):
    """ Process a DataFrame chunk and propagate the errors.

    Args:
        df_chunk (pandas.DataFrame): data containing the Profiles to merge.
        binning (int, optional): binning size. Defaults to 1 (=no binning). No effect if
            method='biglambda'.
        method (str, optional): the processing method. Can be one of
            ['arithmetic mean', 'weighted arithmetic mean', 'circular mean',
            'weighted circular mean', 'arithmetic delta', 'circular delta', 'biglambda'].
            Defaults to 'weighted arithmetic mean'.
        return_V_mats (bool, optional): if set to False, will not return the correlation matrices.
           Doing so saves a lot of memory. Defaults to True.
        cov_mat_max_side (int, optional): maximum size of the covariance matrix, above which it gets
            split and iterated over. Reduce this value in case of memory issues. Defaults to 10000,
            i.e. the matrix will never contain more than 10000 * 10000 elements.

    Returns:
        pandas.DataFrame, dict: the processing outcome, including all the errors,
            and the full correlation matrices (one per uncertainty type) as a dict.

    Note:
        The input format for `df_chunk` is a `pandas.DataFrame` with a very specific structure.
        It requires a single index called `_idx`, with 13 columns per profiles with labels `tdt`,
        `alt`, `val`, 'flg', `ucs`, `uct`, `ucu`, `uc_tot`, `oid`, `mid`, `eid`, and `rid`.
        All these must be grouped together using pd.MultiIndex where the level 0 corresponds
        to the profile number (e.g. 0,1,2...), and the level 1 is the original column name, i.e.::

                       0                                                1
                     alt              tdt    val  ucs  ...   rid alt ...
            _idx
            0      486.7  0 days 00:00:00  284.7  NaN  ...     1 485.8
            1      492.4  0 days 00:00:01  284.6  0.0  ...     1 493.4
            ...

    Note:
        The function will ignore NaNs in a given bin, unless *all* the values in the bin are NaNs.
        See fancy_nansum() for details.

    """

    # Check I have all the required columns
    for col in [PRF_TDT, PRF_ALT, PRF_VAL, PRF_FLG, PRF_UCS, PRF_UCT, PRF_UCU,
                'uc_tot', 'oid', 'mid', 'eid', 'rid']:
        if method in ['biglambda'] and col in [PRF_TDT]:
            continue
        if col not in df_chunk.columns.unique(level=1):
            raise DvasError('Column "{col}" is missing from the DataFrame')

    # Also check that the level 0 starts from 0 onwards
    if 0 not in df_chunk.columns.unique(level=0):
        raise DvasError('Profile "0" is missing ...')

    # How many profiles do I want to combine ?
    n_prf = len(df_chunk.columns.unique(level=0))

    # Data consistency check & cleanup: if val is NaN, all the errors should be NaNs
    # (and vice-versa). Also, if val is NaN, then we should ignore its flags, because as a NaN
    # it will not be taken into account (but this should not trigger a warning).
    for prf_ind in range(n_prf):
        if not all(df_chunk.loc[:, (prf_ind, 'uc_tot')].isna() ==
                   df_chunk.loc[:, (prf_ind, PRF_VAL)].isna()):
            # If I reach this point, then the data is not making sense.
            # Warn the user and clean it up.
            logger.warning("GDP Profile %i: NaN mismatch for 'val' and 'uc_tot'", prf_ind)
            df_chunk.loc[df_chunk.loc[:, (prf_ind, 'uc_tot')].isna().values,
                         (prf_ind, PRF_VAL)] = np.nan
            for col in [PRF_UCS, PRF_UCT, PRF_UCU, 'uc_tot']:
                df_chunk.loc[df_chunk.loc[:, (prf_ind, PRF_VAL)].isna().values,
                             (prf_ind, col)] = np.nan

        if not all(df_chunk.loc[df_chunk.loc[:, (prf_ind, PRF_VAL)].isna(),
                                (prf_ind, PRF_FLG)] == 0):
            logger.debug("GDP profile %i: hiding some flags for the NaN data values.")
            df_chunk.loc[df_chunk.loc[:, (prf_ind, PRF_VAL)].isna().values,
                         (prf_ind, PRF_FLG)] = 0

    # Compute the weights for each point, if applicable
    # First I need to add the new columns (one for each Profile).
    # ----- What follows is problematic because .stack looses the dtype of the flg, which
    # wreaks havoc down the line ---
    # df_chunk = df_chunk.stack(level=0)
    # df_chunk.loc[:, 'w_ps'] = np.nan
    # df_chunk = df_chunk.unstack().swaplevel(0, 1, axis=1).sort_index(axis=1)
    # ------------------------------
    # Crappier (?) alternative, but that doesn't mess up with the column dtypes
    for ind in range(n_prf):
        df_chunk[(ind, 'w_ps')] = np.nan

    # Then I can fill it with the appropriate values
    if 'weighted' in method:
        df_chunk.loc[:, (slice(None), 'w_ps')] = \
            1/df_chunk.loc[:, (slice(None), 'uc_tot')].values**2
    else:
        df_chunk.loc[:, (slice(None), 'w_ps')] = 1.
        # Let us not forget to mask anything that has a NaN value or total error.
        df_chunk.loc[:, (slice(None), 'w_ps')] = df_chunk.loc[:, (slice(None), 'w_ps')].mask(
            df_chunk.loc[:, (slice(None), 'val')].isna().values, np.nan).values
        df_chunk.loc[:, (slice(None), 'w_ps')] = df_chunk.loc[:, (slice(None), 'w_ps')].mask(
            df_chunk.loc[:, (slice(None), 'uc_tot')].isna().values, np.nan).values

    # Combine the Profiles, and compute the Jacobian matrix as well
    if 'arithmetic' in method:
        mode = 'arithmetic'
    elif 'circular' in method:
        mode = 'circular'
    elif method not in ['biglambda']:
        raise DvasError(f'method unknown: {method}')
    else:
        mode = None  # This never gets used, but must be defined for pylint E0606

    if 'mean' in method:
        x_ms, G_mat = weighted_mean(df_chunk, binning=binning, mode=mode)
    elif 'delta' in method:
        x_ms, G_mat = delta(df_chunk, binning=binning, mode=mode)
    elif method == 'biglambda':
        x_ms, G_mat = biglambda(df_chunk)
    else:
        raise DvasError(f'method unknown: {method}')

    # If I was given no data, deal with it.
    if len(df_chunk) == 0:
        return x_ms, None

    # Let's get started with the computation of the errors.
    V_mats = {}  # Will store the covariance matrices in this dict
    # Let us now assemble the U matrices, filling all the cross-correlations for the different
    # types of uncertainties
    for sigma_name in [PRF_UCS, PRF_UCT, PRF_UCU]:
        cc_mat = corr_coeff_matrix(
            sigma_name,
            np.tile(df_chunk.index.values, n_prf),  # step_ids
            # Time-saving trick: make sure we do NOT have an 'object' type !
            oids=df_chunk.xs('oid', level=1, axis=1).T.values.ravel().astype(int),
            mids=df_chunk.xs('mid', level=1, axis=1).T.values.ravel().astype(str),
            rids=df_chunk.xs('rid', level=1, axis=1).T.values.ravel().astype(str),
            eids=df_chunk.xs('eid', level=1, axis=1).T.values.ravel().astype(str),
        )

        # Implement the multiplication. First get the uncertainties ...
        raveled_sigmas = np.array([df_chunk.xs(sigma_name, level=1, axis=1).T.values.ravel()])

        # If all the sigmas are NaN's, let's skip the matrix multiplication to save *a lot* of time
        if np.isnan(raveled_sigmas).all():
            x_ms.loc[:, sigma_name] = np.nan
            V_mats[sigma_name] = np.ma.masked_invalid(np.full((len(x_ms), len(x_ms)), np.nan))
            continue

        # If there are no correlations, I can avoid some potentially large (and time-consuming)
        # matrix multiplications.
        if (np.tril(cc_mat, k=-1) == 0).all() and (np.triu(cc_mat, k=+1) == 0).all():
            variances = (raveled_sigmas**2 * G_mat**2).sum(axis=1).filled(np.nan)
            x_ms.loc[:, sigma_name] = np.sqrt(variances)
            V_mats[sigma_name] = np.ma.masked_invalid(np.diag(variances, k=0))
            continue

        # ... and combine them with the correlation coefficients. Mind the mix of Hadamard and dot
        # products to get the correct mix !
        # Note here that I delay using masked array as long a possible, since these are much slower
        # to handle.

        # Check if I need to break-up the dot products to save memory, or if I can do it all in one
        # step.
        split_into = int(np.ceil(len(df_chunk)/cov_mat_max_side))
        U_mat = np.ma.masked_invalid(np.full(G_mat.T.shape, np.nan))

        # Start looping and splitting as required
        start_ind = 0
        for (ind, sigmas) in enumerate(np.array_split(raveled_sigmas, split_into, axis=1)):

            # How far does this sub-array extend ?
            end_ind = start_ind + sigmas.shape[1]

            # Get the sub U matrix
            sub_U_mat = np.dot(sigmas.T, raveled_sigmas)

            # Deal with correlation coefficients if required
            if not (cc_mat[start_ind:end_ind] == 1).all():
                np.multiply(cc_mat[start_ind:end_ind], sub_U_mat, out=sub_U_mat)

            sub_U_mat = np.ma.masked_invalid(sub_U_mat, copy=False)

            # Let's compute the covariance matrix for the merged profile (for the specific
            # error type).
            # This is a square matrix, with the off-axis elements containing the covarience terms
            # for the merged profile. All these matrices are masked arrays, such that bad values
            # will be correctly ignored .... unless that is all we have for a given bin.
            U_mat[start_ind:end_ind] = np.ma.dot(sub_U_mat, G_mat.T)

            # Let's not forget to update the starting index for the next loop
            start_ind += sigmas.shape[1]

        # Do the second part of the GUG dot product. I won't split this one in chunks.
        U_mat = np.ma.dot(G_mat, U_mat)

        # Assign the propagated uncertainty values to the combined df_chunk, taking care of
        # replacing any masked element with NaNs.
        x_ms.loc[:, sigma_name] = np.sqrt(U_mat.diagonal().filled(np.nan))

        # Keep track of the covariance matrix, in order to return them all to the user.
        # Unless I am trying to keep the memory use as low as possible.
        if return_V_mats:
            V_mats[sigma_name] = U_mat
        else:
            V_mats[sigma_name] = None

    # All done
    return x_ms, V_mats
