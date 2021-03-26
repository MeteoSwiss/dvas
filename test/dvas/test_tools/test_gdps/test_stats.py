# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing 'stats' classes and function of the tools.gdps submodule.

"""

# Import from python packages and modules
import numpy as np
import pytest
import pandas as pd

from dvas.data.strategy.data import GDPProfile
from dvas.data.data import MultiGDPProfile
from dvas.database.database import InfoManager

# Functions to test
from dvas.tools.gdps.stats import get_incompatibility

# Define db_data. This is some black magic that is directly related to conftest.py
# This is a temporary db, that is required for get_info() to work properly with mdl_id.
# It all relies on the db config files located in the processing arena
db_data = {
    'sub_dir': 'test_tool_stats',
    'data': [{'mdl_name': 'AR-GDP_001',
              'srn': 'AR1',
              'pid': '0',},
             {'mdl_name': 'BR-GDP_001',
              'srn': 'BR1',
              'pid': '1',},
            ]}

@pytest.fixture
def gdp_2_prfs(db_init):
    """ Return a MultiGDPProfile with 2 GDPprofiles in it. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_1 = InfoManager('20210302T0000Z', oids[0], tags=['e:1', 'r:1'])
    data_1 = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10., 20., 30.], 'flg': [1, 1, 1],
                           'tdt': [1e9, 2e9, 3e9], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})
    info_2 = InfoManager('20210302T0000Z', oids[1], tags=['e:1', 'r:1'])
    data_2 = pd.DataFrame({'alt': [11., 16., 20.1], 'val': [10.5, 21., np.nan], 'flg': [1, 1, 1],
                           'tdt': [1e9, 2e9, 3e9], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [GDPProfile(info_1, data_1), GDPProfile(info_2, data_2)])

    return multiprf

def test_get_incompatibility(gdp_2_prfs):
    """ Test the higher-level incompatibility checks """

    out = get_incompatibility(gdp_2_prfs, alpha=0.0027, bin_sizes=[1], do_plot=False)

    import pdb
    pdb.set_trace()

    assert isinstance(out, dict)
