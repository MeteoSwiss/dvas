# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains WMO-related tools.

"""

import numpy as np


def geom2geopot(vals, lat):
    """ Convert geometric altitudes to geopotential heights.

    Args:
        vals (ndaray or pd.Series or pd.DataFrame): geometric altitudes
        lat (float): geodetic latitude, in radians

    Uses the Mahoney equations from the CIMO guide.

    Reference:

        WMO GUIDE TO METEOROLOGICAL INSTRUMENTS AND METHODS OF OBSERVATION (the CIMO Guide),
        WMO-No. 8 (2014 edition, updated in 2017), Part I - MEASUREMENT OF METEOROLOGICAL VARIABLES,
        Ch. 12 - Measurement of upper-air pressure, temperature, humidity,
        Sec. 12.3.6 - Use of geometric height observations instead of pressure sensor observations,
        p. 364.

     """

    gamma_45 = 9.80665

    R_lat = 6378.137 / (1.006803 - 0.006706 * np.sin(lat)**2)
    R_lat *= 1e3  # Convert to m

    gamma_s = (1 + 0.00193185 * np.sin(lat)**2)
    gamma_s /= (1 - 0.00669435 * np.sin(lat)**2)**0.5
    gamma_s *= 9.780325

    return gamma_s/gamma_45 * R_lat * vals / (R_lat + vals)
