"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Specialized mathematical operation on data

"""

# Import external packages and modules
import numpy as np
import pandas as pd


def crosscorr(datax, datay, lag=0, wrap=False, method='kendall'):
    """Lag-N cross correlation.
    If wrap is False, shifted data are filled with NaNs.

    Args:
        datax, datay (pandas.Series): Must be of equal length
        lag(`obj`:int, optional):  Default is 0
        wrap (`obj`:bool, optional:  (Default value = False)
        method (`obj`:str, optional): kendall, pearson, spearman.
            Default is 'kendall'

    Returns:
        float

    """

    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        out = datax.corr(shiftedy, method=method)
    else:
        out = datax.corr(datay.shift(lag), method=method)

    return out


def calc_tropopause(temp, alt, start_alt=5000, layer=2000, grad_cond=-2):
    """Calculate tropopause altitude

    Note:
        From the WMO definition, the tropopause is defined as the lowest
        altitude where the vertical thermal gradient becomes less than -2 K per
        km in a layer at least 2 km thick.

    Source:
        G. Romanens, MeteoSwiss

    Args:
        temp (pd.Series): Temperature profile (°C).
        alt (pd.Series): Altitude profile in (m).
        start_alt (float): Starting search altitude (m). Default to 5000.
        layer (flaot): Layer in which the thermal gradient is checked (m).
            Default to 2000.
        grad_cond (float): Minimal allowed thermal gradient (°C). Default to -2.

    Returns:
        pd.index: Tropopause index

    """

    # Define
    alt.name = 'alt'
    temp.name = 'temp'
    data = pd.concat([alt, temp], axis=1)
    idxmin = None

    # Loop on data
    for idxmin, row in data[data['alt'] > start_alt].iterrows():

        # Found max layer level (idxmin + layer thickness)
        idxmax = data.loc[idxmin:].loc[
            (data.loc[idxmin:, 'alt'] - row['alt']) < layer
        ].index[-1]

        # Calculate minimal thermal gradient in layer
        min_grad = np.nanmin(
            (
                (row['temp'] - data['temp'].loc[idxmin:idxmax]) /
                (row['alt'] - data['alt'].loc[idxmin:idxmax])
            ) * 1000
        )

        # Break if minimal gradient in layer > gradient condition
        if min_grad > grad_cond:
            break

    # Test found index
    if idxmin is None:
        raise TropopauseError('Profile minimal altitude criteria is not fulfilled')
    elif idxmin == data.index[-1]:
        raise TropopauseError('Thermal gradient criteria is not fulfilled')

    return idxmin


class TropopauseError(Exception):
    """Exception in no tropopause where found"""


def calc_tropopause_old(alti, press, temp):
    """Calculate tropopause altitude

    Note:
        From the WMO definition, the tropopause is defined as the lowest
        altitude where the vertical thermal gradient becomes less than -2 K per
        km in a layer at least 2 km thick.

    Source:
        G. Romanens, MeteoSwiss

    Args:
        mydate:
        alti:
        press:
        temp:

    Returns:

    """
    Tropo = {}
    Tropo['Alti'] = np.nan
    Tropo['Press'] = np.nan
    Tropo['Temp'] = np.nan
    indminlevel = np.nan
    indmaxlevel = np.nan

    for i in range(0,len(press)):
        if (press.iloc[i]< 500):
            indminlevel = i
            break

    if (np.isnan(indminlevel)):
        return (Tropo)

    for i in range(indminlevel, len(alti)):
        indmaxlevel = 0
        for j in range(i, len(alti)):
            if alti.iloc[i] + 2000 < alti.iloc[j]:
                indmaxlevel = j - 1
                break

        if indmaxlevel == 0: break

        candidate = True
        while True:
            if indmaxlevel == i: break
            deltaA = alti.iloc[i] - alti.iloc[indmaxlevel]
            deltaT = temp.iloc[i] - temp.iloc[indmaxlevel]

            try:
                grad = deltaT * 1000 / deltaA
            except ZeroDivisionError:
                indmaxlevel = indmaxlevel - 1
                continue

            if grad < -2:
                candidate = False
                break
            indmaxlevel = indmaxlevel - 1

        if candidate == True:
            Tropo['Alti'] = alti.iloc[i]
            Tropo['Press'] = press.iloc[i]
            Tropo['Temp'] = temp.iloc[i]

            break

    return(Tropo)
