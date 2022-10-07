# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing 'tools' classes and function of the wmo submodule.

"""

import numpy as np

from dvas.tools import wmo


def test_geom2geopot():
    """ Test the WMO-CIMO conversion """

    # Table 12.4 in the CIMO guide
    cimo_vals = {8000: 25, 16000: 70, 24000: 135, 32000: 220}

    for (val, offset) in cimo_vals.items():
        delta = val - wmo.geom2geopot(val, np.radians(22))
        delta = delta - delta % 5
        assert delta == offset
