"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: UAII2022 recipe tools
"""

# Import from Python
import logging
from copy import deepcopy
import numpy as np
from scipy import interpolate

# Import from dvas
from dvas.logger import log_func_call
from dvas.hardcoded import PRF_TDT, PRF_ALT, PRF_VAL

# Import from this module
from ..errors import DvasRecipesError

# Setup local logger
logger = logging.getLogger(__name__)


def get_query_filter(tags_in: list = None, tags_in_or: list = None,
                     tags_out: list = None, mids: list = None,
                     oids: list = None) -> str:
    """ Assembles a str to query the dvas DB, given a list of tags to include and/or exclude.

    Args:
        tags_in (list, optional): list of tags required to be present (AND)
        tags_in_or (list, optional): list of tags required to be present (OR)
        tags_out (list, optional): list of tags required to be absent
        mids (list, optional): list of mids required (OR)
        oids (list, optional): list of oids required (OR)
        tods (list, optional): list of times-of-day to look for (OR)

    Returns:
        str: the query filter
    """

    filt = []

    if tags_in is not None:
        filt += ["tags('" + "'), tags('".join(tags_in) + "')"]

    if tags_in_or is not None:
        filt += ["or_(tags('" + "'), tags('".join(tags_in_or) + "'))"]

    if tags_out is not None:
        filt += ["not_(tags('" + "')), not_(tags('".join(tags_out) + "'))"]

    if mids is not None:
        filt += ["or_(mid('" + "'), mid('".join(mids) + "'))"]

    if oids is not None:
        filt += ["or_(oid(" + "), oid(".join([str(item) for item in oids]) + "))"]

    if len(filt) == 0:
        return ''

    return 'and_(' + ', '.join(filt) + ')'


@log_func_call(logger, time_it=False)
def find_tropopause(rs_prf, min_alt=5500, algo='gruan'):
    """ Find the tropopause altitude in a given RSProfile.

    Args:
        rs_prf (RSProfile): the temperature profile from which to derive the tropopause height.
        min_alt (int| float, optional): minimum altitude above which to look for the tropopause.
            Defaults to 5000 m.
        algo (str, optional): which algorithm to use, one of ['wmo1957', 'mch', 'gruan'].
            Defaults to 'gruan'. See below for details.

    Returns:
        float, float, float: the troposphere index, altitude, and timestep.

    Note:
        The WMO definition of the tropopause(s) is as follows:

        " (a) The first tropopause is defined as the lowest level at which the lapse rate
        decreases to 2degC/km or less, provided also the average lapse rate between this level
        and all higher levels within 2 km does not exceed 2degC/km

        (b) If above the first tropopause the average lapse rate between any level and all
        higher levels within 1 km exceeds 3degC/km, then a second tropopause is defined by the
        same criterion as under (a). This tropopause may be either within or above the 1 km layer."

        Source: World Meteorological Organization (1957),
        Meteorology - A three-dimensional science:
        Second session of the Commission for Aerology, WMO Bulletin, vol. IV, no. 4,
        https://library.wmo.int/doc_num.php?explnum_id=6960

        The lapse rate is the decrease of an atmospheric variable with height,
        the variable being temperature unless otherwise specified.

        Typically, the lapse rate is the negative of the rate of temperature change with
        altitude change.

    The 1957 WMO definition (`algo='wmo1957'`) of the tropopause is problematic for high resolution
    profile data, as it leads to a detection of the tropopause several hundred meters below the
    "knee" of the profile. This behavior is caused by the averaging of all subsequent levels, which
    allows for a significant number of them to fail the lapse rate criteria if enough others do.

    The `algo='mch'` alternative defines the first tropopause as the lowest level at which the lapse
    rate decreases to 2deg/km or less, provided also ALL the lapse rateS between this level and all
    higher levels within 2km do not exceed 2deg/km.

    The `algo='gruan'` alternative is used to compute the tropopause in GRUAN Data Products,
    including the RS41. It defines the first tropopause as the lowest level at which the lapse
    rate decreases to 2deg/km or less, provided also ALL the MEAN lapse rateS (between this level
    and all higher levels, COMPUTED FROM ALL SEQUENTIAL LEVEL PAIRS LOCATED WITHIN A GIVEN LEVEL
    INTERVAL) within 2 km do not exceed 2deg/km.

    """

    # Let's make a deepcopy of the DataFrame, to avoid messing up with the user input
    rs_prf = deepcopy(rs_prf)

    # TODO: in this function, we always assume that we have meters ...
    # This should (at the very least) be checked for.
    # This would most likely require #228 to be fixed, though ...

    # Let us duplicate the alt index as a column ...
    rs_prf.data.loc[:, PRF_ALT] = rs_prf.data.index.get_level_values('alt').values

    # Holes inside the profile are problematic, as they may lead to erroneous detections
    # To avoid these, let's interpolate linearly over them. No interpolated value can be the
    # tropopause, but at least this will avoid faulty detections.
    # Here, I use scipy for the interpolation to do it as a function of time, as pandas is not
    # great at doing it with a MultiIndex.
    x = rs_prf.data.index.get_level_values(PRF_TDT).total_seconds().values
    y = rs_prf.data.loc[:, PRF_VAL].values
    rs_prf.data.loc[:, 'temp_interp'] = interpolate.interp1d(x[~np.isnan(y)], y[~np.isnan(y)],
                                                             kind='linear', fill_value=np.nan,
                                                             bounds_error=False)(x)
    y = rs_prf.data.loc[:, PRF_ALT].values
    rs_prf.data.loc[:, 'alt_interp'] = interpolate.interp1d(x[~np.isnan(y)], y[~np.isnan(y)],
                                                            kind='linear', fill_value=np.nan,
                                                            bounds_error=False)(x)

    # ... so we can compute the lapse rate
    rs_prf_diff = rs_prf.data.diff()
    rs_prf.data.loc[:, 'lapse_rate'] = - rs_prf_diff.temp_interp / rs_prf_diff.alt_interp * 1e3

    # Loop through all altitudes, stopping only were the lapse rate is small enough (and valid)
    for idxmin, row in rs_prf.data[rs_prf.data[PRF_ALT] > min_alt].iterrows():
        if np.isnan(row[PRF_VAL]):
            # I refuse to detect the tropopause at an interpolated location.
            continue
        if row['lapse_rate'] > 2:
            continue

        # Let's extract a 2km-thick layer above ...
        cond = ((rs_prf.data.loc[:, 'alt_interp'] - row['alt_interp']) / 1000) <= 2
        cond *= ((rs_prf.data.loc[:, 'alt_interp'] - row['alt_interp']) / 1000) > 0
        layer = rs_prf.data.loc[cond].copy(deep=True)

        # ... and compute the lapse rate with respect to its base
        if algo in ['wmo1957', 'mch']:
            layer.loc[:, 'lay_lapse_rate'] = - (layer.loc[:, 'temp_interp']-row['temp_interp']) / \
                (layer.loc[:, 'alt_interp']-row['alt_interp']) * 1e3

        if algo == 'gruan':
            layer.loc[:, 'lay_lapse_rate'] = \
                layer.lapse_rate.cumsum().values / np.arange(1, len(layer)+1, 1)

        # Is this the tropopause ?
        if algo == 'wmo1957' and layer.lay_lapse_rate.mean(skipna=True) <= 2:
            return idxmin
        # ... but for reasons unclear, the original MCH codes use the following, which also
        #  better matches the GRUAN values.
        if algo == 'mch' and (layer.lay_lapse_rate[layer.lay_lapse_rate.notna()] <= 2).all():
            return idxmin
        # Here is the actual GRUAN condition, applied on the lapse rates between steps
        if algo == 'gruan' and layer.lay_lapse_rate.max(skipna=True) <= 2:
            return idxmin

    raise DvasRecipesError('No tropopause found !')
