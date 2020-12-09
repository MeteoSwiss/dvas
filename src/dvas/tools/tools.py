# -*- coding: utf-8 -*-
"""

Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains low-level, stand-alone, dvas tools.

"""

# Import from Python packages
import numpy as np
import pandas as pd

# Import from this package
from ..dvas_logger import log_func_call, dvasError
from ..dvas_logger import tools_logger as logger

@log_func_call(logger)
def fancy_nansum(vals, axis=None):
    """ A custom nansum routine that treats NaNs as zeros, unless the data contains *only* NaNs,
    if which case it returns a NaN.

    Args:
        vals (pandas.DataFrame): the data to sum.
        axis (int, optional): on which axis to run the fancy nansum. Defaults to None
            (=sum everything).

    Returns:
        float: the nansum(), or nan if the data contains only nans.

    Example::

        In: vals = pd.DataFrame(np.ones((4,3)))
        In: vals.iloc[0] = np.nan
        In: vals[0][1] = np.nan
        In: vals[1][1] = np.nan
        In: vals[0][2] = np.nan
        In: vals
        0    1    2
        0  NaN  NaN  NaN
        1  NaN  NaN  1.0
        2  NaN  1.0  1.0
        3  1.0  1.0  1.0

        In: fancy_nansum(vals)
        6.0

        In: fancy_nansum(vals, axis=1)
        0    NaN
        1    1.0
        2    2.0
        3    3.0
        dtype: float64

        In: vals.sum(skipna=True)
        0    0.0
        1    1.0
        2    2.0
        3    3.0
        dtype: float64

    """

    # If I get a Series instead of a DataFrame, deal with. This happens, for example, when using
    # .groupby().aggregate(fancy_nansum) ...
    if isinstance(vals, pd.core.series.Series):
        vals = pd.DataFrame(vals)

    # Check the data type to make sure it is what I expect
    if not isinstance(vals, pd.core.frame.DataFrame):
        raise dvasError("vals should be pandas DataFrame, not: %s" % (type(vals)))

    # If no axis is specified, let us just sum every element
    if axis is None:
        if np.all(vals.isna()):
            return np.nan

        return np.nansum(vals.values)

    return vals.sum(axis=axis, skipna=True).mask(vals.isna().all(axis=axis), np.nan)


@log_func_call(logger)
def weighted_mean(vals, weights, binning=1):
    """ Compute the weighted mean of the columns of a pd.DataFrame, given weights specified in a
    separate DataFrame.

    Args:
        vals (pands.DataFrame): 2-D data to weighted-average on a row-per-row basis.
        weights (pands.DataFrame): 2-D weights, with the same shape as vals.

    Returns:
        (pandas.DataFrame, list of list): weighted mean, and non-zero Jacobian matrix elements,
        intended for error propagation purposes.

    Instead of computing a full Jacobian matrix (which is mostly filled with 0's and seriously
    slows the code down), this function only computes its non-zero elements and feed them back to
    the user. This is not the most ideal, as the user needs to understand what is happening ... and
    which elements are being fed back ... but it is soooooooo much faster than carrying O(1e8) 0's
    around !

    Note:
        - The function will ignore NaNs, unless *all* the values in a given bin are NaNs. See
          fancy_nansum() for details.
        - The function is *significantly* slower if binning > 1, because of a .groupby().aggregate()
          function call.

    """

    # First, some sanity checks
    if np.shape(vals) != np.shape(weights):
        raise dvasError("vals and weights must have the same shape.")
    if vals.ndim != 2:
        raise dvasError("vals and weights must be 2-D.")

    # Force the weights to be NaNs if the data is a NaN. Else, the normalization will be off.
    weights[vals.isna()] = np.nan

    # Compute the val * weights DataFrame
    wx_ps = vals * weights

    # Sum val * weight and weights accross profiles.
    # Note the special treatment of NaNs: ignored, unless that is all I get from all the profiles at
    # a given time step/altitude.
    wx_s = fancy_nansum(wx_ps, axis=1)
    w_s = fancy_nansum(weights, axis=1)

    # Then sum these across the time/altitude layers according to the binning
    if binning > 1:
        # Note again the special treatment of NaNs:
        # ignored, unless that is all I have in a given bin.
        wx_ms = wx_s.groupby(wx_s.index//binning).aggregate(fancy_nansum)
        w_ms = w_s.groupby(w_s.index//binning).aggregate(fancy_nansum)
    else:
        # If no binning is required, then do nothing and save *a lot* of time.
        wx_ms = wx_s
        w_ms = w_s

    # If any of the total weight is 0, replace it with nan's (to avoid runtime Warnings).
    w_ms.mask(w_ms == 0, np.nan, inplace=True)

    # Compute the weighted mean
    # To avoid some runtime Warning, replace any 0 weight with nan's
    x_ms = wx_ms / w_ms

    """ The code below is mathematically better, but *very* slow.
    #
    # What follows computes the FULL Jacobian matrix. This is slow, as the size of the matrix
    # is n_gdp*len(vals)*len(vals)/binning ~ 3x7000x7000 = O(1e8) elements.
    # The vast majority of these elements are exactly 0, especially with small/no binning.

    # First, the full matrix of weights
    # Mind the "T" to keep things in order !
    w_mat = np.tile(weights.values.T.flatten(), (len(x_ms), 1))
    # Next we need to which weights are used for each level of the final (binned) profile
    # (the matrix should be mostly filled with 0)
    # This is slow
    #in_lvl = np.array([vals.shape[1]*[1 if j//binning == i else 0 for j in range(len(vals))]
    #                   for i in range(len(x_ms))])
    # This is better
    #in_lvl = np.array([vals.shape[1]*([0]*i*binning + [1]*binning + [0]*(len(vals)-1-i)*binning)
    #                   for i in range(len(x_ms))])
    # Apply the selection to the weight matrix. We avoid the multiplication to properly deal with
    # NaNs.
    #w_mat[in_lvl == 0] = 0

    # This is even better
    rows, cols = np.indices((len(x_ms), vals.size))
    w_mat[(cols % len(vals))//binning != rows] = 0

    # I also need to assemble a mtrix of the total weights for each (final) row.
    wtot_mat = np.tile(w_ms.values, (vals.size, 1)).T

    # I can now assemble the Jacobian
    jac_mat = w_mat / wtot_mat
    """

    # Instead of computing a full Jacobian matrix (which is mostly filled with 0's and seriously
    # slows the code down), let's only compute its non-zero elements and feed them back to the user.
    # This is not the most ideal, as the user needs to understand what is happening ...
    # but it is soooooooo much faster than carrying O(1e8) 0's around !

    # First, let's get the flatten weight array's this has  a length of n_prf * len(prf)
    # ~3x7000 = 21000 values.
    w_flat = weights.values.T.flatten()
    w_ind = np.indices([len(w_flat)])[0] % len(vals)

    # Next build a list of list, where each element contains the non-zero Jacobian elements.
    jac_elmt = [w_flat[w_ind//binning == i]/w_tot
                for i, w_tot in enumerate(w_ms.mask(w_ms == 0, np.nan))]

    return x_ms, jac_elmt

@log_func_call(logger)
def delta(vals, binning=1):
    """ Compute the delta of a 2-columns pd.DataFrame (i.e. col1 - col2).

    Args:
        vals (pandas.DataFrame): 2-D data with 2 columns.

    Returns:
        (pandas.DataFrame, list of list): weighted mean, and list of non-zero Jacobian matrix
        elements, intended for error propagation purposes.

    Instead of computing a full Jacobian matrix (which is mostly filled with 0's and seriously
    slows the code down), this function only computes its non-zero elements and feed them back to
    the user. This is not the most ideal, as the user needs to understand what is happening ... and
    which elements are being fed back ... but it is soooooooo much faster than carrying O(1e8) 0's
    around !

    Note:
        The function will ignore NaNs, unless *all* the values in a given bin are NaNs.
        See fancy_nansum() for details.

    """

    # First, some sanity checks
    if vals.shape[1] != 2:
        raise dvasError("vals should have 2 columns only.")
    if vals.ndim != 2:
        raise dvasError("vals must be 2-D.")

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
