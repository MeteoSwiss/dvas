# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing 'gruan' classes and function of the tools submodule.

"""

# Import from python packages and modules
import numpy as np
import pytest
import pandas as pd

from dvas.data.strategy.data import GDPProfile
from dvas.data.data import MultiGDPProfile
from dvas.database.database import InfoManager

# Functions to test
from dvas.tools.gdps import combine

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
def gdp_1_prfs(db_init):
    """ Return a MultiGDPProfile with 1 GDPprofiles in it. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_1 = InfoManager('20210302T0000Z', oids[0], tags=['e:1', 'r:1'])
    data_1 = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10., 20., 30.], 'flg': [1, 1, 1],
                           'tdt': [1, 2, 3], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1], 'uct': [1, 1, 1],
                           'ucu': [1, 1, 1]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [GDPProfile(info_1, data_1)])

    return multiprf

@pytest.fixture
def gdp_2_prfs(db_init):
    """ Return a MultiGDPProfile with 2 GDPprofiles in it. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_1 = InfoManager('20210302T0000Z', oids[0], tags=['e:1', 'r:1'])
    data_1 = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10., 20., 30.], 'flg': [1, 1, 1],
                           'tdt': [1, 2, 3], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1], 'uct': [1, 1, 1],
                           'ucu': [1, 1, 1]})
    info_2 = InfoManager('20210302T0000Z', oids[1], tags=['e:1', 'r:1'])
    data_2 = pd.DataFrame({'alt': [11., 15., 20.1], 'val': [10., 21., np.nan], 'flg': [1, 1, 1],
                           'tdt': [1, 2, 3], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1], 'uct': [1, 1, 1],
                           'ucu': [1, 1, 1]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [GDPProfile(info_1, data_1), GDPProfile(info_2, data_2)])

    return multiprf

# Let us test a series of conditions for the different types of uncertainty types
def test_combine(gdp_1_prfs, gdp_2_prfs):
    """Function used to test if the routine combining GDP profiles is ok.

    The function tests:
        - correct propagation of errors when rebining a single profile

    Note:
       Inspired, in part, from `this post
       <https://stackoverflow.com/questions/39896716/can-i-perform-multiple-assertions-in-pytest>`__
       by Jon Clements.

    """

    # 0) Ultra-basic test: the mean of a single profile with binning = 1 should return the
    # same thing

    for method in ['mean', 'weighted mean']:
        out = combine(gdp_1_prfs, binning=1, method=method, chunk_size=200, n_cpus='max')

        import pdb
        pdb.set_trace()

        for key in ['val', 'ucr', 'ucs', 'uct', 'ucu']:
            assert np.array_equal(out.profiles[0].data[key], gdp_1_prfs.profiles[0].data[key])
        for key in ['tdt', 'alt']:
            assert np.array_equal(out.profiles[0].data.index.get_level_values(key),
                                  gdp_1_prfs.profiles[0].data.index.get_level_values(key))
