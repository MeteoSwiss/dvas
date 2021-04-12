"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.plots.plots module.

"""

# Import from python packages and modules
import pytest
import numpy as np
import pandas as pd

# Import from current package
from dvas.data.strategy.data import GDPProfile
from dvas.data.data import MultiGDPProfile
from dvas.database.database import InfoManager

# Functions to test
#import dvas.plots.plots as dpp

# Define db_data. This is some black magic that is directly related to conftest.py
# This is a temporary db, that is required for get_info() to work properly with mdl_id.
# It all relies on the db config files located in the processing arena
db_data = {
    'sub_dir': 'test_tool_gdps',
    'data': [{'mdl_name': 'AR-GDP_001',
              'srn': 'AR1',
              'pid': '0',},
             {'mdl_name': 'BR-GDP_001',
              'srn': 'BR1',
              'pid': '1',},
             {'mdl_name': 'BR-GDP_001',
              'srn': 'CR1',
              'pid': '2',},
            ]}

@pytest.fixture
def gdp_3_prfs(db_init):
    """ Return a MultiGDPProfile with 3 GDPprofiles in it. """

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
    info_3 = InfoManager('20210302T0000Z', oids[2], tags=['e:1', 'r:1'])
    data_3 = pd.DataFrame({'alt': [10.1, 17., 20.], 'val': [11., 21.1, np.nan], 'flg': [1, 1, 1],
                           'tdt': [1e9, 2e9, 3e9], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [GDPProfile(info_1, data_1), GDPProfile(info_2, data_2),
                     GDPProfile(info_3, data_3)])

    return multiprf

#def test_multiprf(gdp_3_prfs):
#    """ Test the multiprf plotting routine """
#
#    dpp.multiprf(gdp_3_prfs, index='alt', label='oid', uc=None, show=True)
