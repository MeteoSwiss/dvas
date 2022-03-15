# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains low-level, stand-alone, dvas tools.

"""

# Import from Python packages
import logging
import numpy as np
import numpy.ma as ma
import pandas as pd

# Import from this package
from ..errors import DvasError
from ..logger import log_func_call

# Setup local logger
logger = logging.getLogger(__name__)


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
        raise DvasError("vals should be pandas DataFrame, not: %s" % (type(vals)))

    # If no axis is specified, let us just sum every element
    if axis is None:
        if np.all(vals.isna()):
            return np.nan

        return np.nansum(vals.values)

    return vals.sum(axis=axis, skipna=True).mask(vals.isna().all(axis=axis))


@log_func_call(logger)
def fancy_bitwise_or(vals, axis=None):
    """ A custom bitwise_or routine that ignores NaN unless only NaNs are provided, in which case
    a NaN is returned.

    Args:
        vals (pandas.DataFrame): the data to sum.
        axis (int, optional): on which axis to run the fancy nansum. Defaults to None
            (=sum everything).

    Returns:
        pd.NA|int|pd.array: the result as a scale if axis=None, and a pandas array with dtype Int64
            if not.
    """

    # Let's make sure I have been given integers, in order to run bitwise operations
    if not all(dtps := [pd.api.types.is_integer_dtype(item) for item in vals.dtypes]):
        raise DvasError('Ouch ! I need ints to perform a bitwise OR, but I got:', dtps)

    # Easy if all is nan and axis is None
    if axis is None and vals.isna().all(axis=axis):
        return pd.NA  # Return the 'Int64' NaN value

    # Let's compute a bitwise_or along the proper axis, not forgetting to mask any NA to not blow
    # things up.
    out = np.bitwise_or.reduce(vals.mask(vals.isna(), 0), axis=axis)

    # If the axis is None, I'll be returning a scalar, so I do not need to do anything else
    if axis is None:
        return out

    # If the axis is not None, then I need to mask the rows or columns that are full of NA
    # Note here that out is a numpy array with dtype "object". We shall convert this to a pandas
    # array with dtype='Int64'
    out[vals.isna().all(axis=axis)] = pd.NA
    return pd.array(out).astype('Int64')


@log_func_call(logger)
def df_to_chunks(df, chunk_size):
    """ A utility function that breaks a Pandas dataframe into chunks of a specified size.

    Args:
        df (pandas.DataFrame): the pandas DataFrame to break-up.
        chunk_size (int): the length of each chunk. If len(pdf) % chunk_size !=0, the last chunk
            will be smaller than the other ones.

    Returns:
        list of pandas.DataFrame: the ordered pieces of the original DataFrame
    """

    # How many chunks does this correspond to ?
    n_chunks = int(np.ceil(len(df)/chunk_size))

    # Here, I rely on the fact that .iloc will not wrap around if the upper limit goes beyond the
    # end of the array. I.e. it automatically crops the last chunk where the Profile stops.
    # Note the .copy(), which is essential to avoid SettingWithCopyWarning issues down the line ...
    return [df.iloc[chunk_id * chunk_size: (chunk_id+1) * chunk_size].copy()
            for chunk_id in range(n_chunks)]
