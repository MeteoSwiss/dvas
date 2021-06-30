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
from dvas.tools.gdps.stats import ks_test, gdp_incompatibilities, gdp_validities

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
             {'mdl_name': 'BR-GDP_001',
              'srn': 'BR2',
              'pid': '1',},
            ]}

@pytest.fixture
def gdp_2_prfs(db_init):
    """ Returns a MultiGDPProfile with 2 GDPprofiles in it. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_1 = InfoManager('20210302T0000Z', oids[0], tags=['e:1', 'r:1'])
    data_1 = pd.DataFrame({'alt': [5, 10., 15., 20.], 'val': [0, 10., 20., 30.],
                           'flg': [1, 1, 1, 1], 'tdt': [0e9, 1e9, 2e9, 3e9],
                           'ucr': [1, 1, 1, 1], 'ucs': [1, 1, 1, 1],
                           'uct': [1, 1, 1, 1], 'ucu': [1, 1, 1, 1]})
    info_2 = InfoManager('20210302T0000Z', oids[1], tags=['e:1', 'r:1'])
    data_2 = pd.DataFrame({'alt': [5, 11., 16., 20.1], 'val': [2, 14., 26., np.nan],
                           'flg': [1, 1, 1, 1], 'tdt': [0e9, 1e9, 2e9, 3e9],
                           'ucr': [1, 1, 1, 1], 'ucs': [1, 1, 1, 1],
                           'uct': [1, 1, 1, 1], 'ucu': [1, 1, 1, 1]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [GDPProfile(info_1, data_1), GDPProfile(info_2, data_2)])

    return multiprf

@pytest.fixture
def gdp_3_prfs(db_init):
    """ Returns a MultProfile with 3 GDPProfile in it. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_1 = InfoManager('20210302T0000Z', oids[0], tags=['e:1', 'r:1'])
    data_1 = pd.DataFrame({'alt': [5, 10., 15., 20.], 'val': [0, 10., 20., 30.],
                           'flg': [1, 1, 1, 1], 'tdt': [0e9, 1e9, 2e9, 3e9],
                           'ucr': [1, 1, 1, 1], 'ucs': [1, 1, 1, 1],
                           'uct': [1, 1, 1, 1], 'ucu': [1, 1, 1, 1]})
    info_2 = InfoManager('20210302T0000Z', oids[1], tags=['e:1', 'r:1'])
    data_2 = pd.DataFrame({'alt': [5, 11., 16., 20.1], 'val': [2, 14., 26., np.nan],
                           'flg': [1, 1, 1, 1], 'tdt': [0e9, 1e9, 2e9, 3e9],
                           'ucr': [1, 1, 1, 1], 'ucs': [1, 1, 1, 1],
                           'uct': [1, 1, 1, 1], 'ucu': [1, 1, 1, 1]})
    info_3 = InfoManager('20210302T0000Z', oids[2], tags=['e:1', 'r:1'])
    data_3 = pd.DataFrame({'alt': [5, 11., 16., 20.1], 'val': [2, 14., 26., np.nan],
                           'flg': [1, 1, 1, 1], 'tdt': [0e9, 1e9, 2e9, 3e9],
                           'ucr': [1, 1, 1, 1], 'ucs': [1, 1, 1, 1],
                           'uct': [1, 1, 1, 1], 'ucu': [1, 1, 1, 1]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [GDPProfile(info_1, data_1), GDPProfile(info_2, data_2),
                     GDPProfile(info_3, data_3)])

    return multiprf

def test_ks_test(gdp_2_prfs):
    """ Test the proper usage of the KS test """

    out_1 = ks_test(gdp_2_prfs, alpha=0.0027, m_val=1, n_cpus=1)
    assert len(out_1) == len(gdp_2_prfs[0]) # Correct length ?
    assert round(100*(1-out_1.loc[0, 'p_ksi']), 1) == 68.3 # 1-sigma
    assert round(100*(1-out_1.loc[1, 'p_ksi']), 1) == 95.4 # 2-sigma
    assert round(100*(1-out_1.loc[2, 'p_ksi']), 1) == 99.7 # 3-sigma
    assert out_1.isna().loc[3, 'p_ksi'] # Bad point
    assert all(out_1.loc[:2, 'f_pqi'].values == [0, 0, 1])  # Correct flags
    assert out_1.isna().loc[3, 'f_pqi']

    # Now with some binning
    out_2 = ks_test(gdp_2_prfs, alpha=0.0027, m_val=2, n_cpus=1)
    assert len(out_2) == len(gdp_2_prfs[0])//2 + len(gdp_2_prfs[0])%2 # Correct length ?

    # Partial NaN's get ignored completely ?
    assert out_1.loc[2, 'k_pqi'] == out_2.loc[1, 'k_pqi']


def test_gdp_incompatibilities(gdp_2_prfs):
    """ Test the higher-level incompatibility checks """

    out = gdp_incompatibilities(gdp_2_prfs, alpha=0.0027, m_vals=[1, 2], do_plot=True)
    assert isinstance(out, pd.DataFrame)

def test_gdp_validities(gdp_3_prfs):
    """ Test the different validation strategies to determine if GDPs are compatible with each
    others. """

    # First, assemble an incompatibility list, assuming 3 profiles, and 2 binning levels
    # I will run the gdp_incompatibilities function to get the proper structure I need ...
    incompat = gdp_incompatibilities(gdp_3_prfs, m_vals=[1, 2, 3], do_plot=False)
    # ... end then replace it with data I like better, to not be dependant on the alpha lvl, etc ...

    # ... before feeding it into the function to test
    out = gdp_validities(incompat, m_vals=[1], strategy='all-or-none')
    assert isinstance(out, pd.DataFrame)
    assert out.all(axis=1)[0] # 1-sigma - all-or-none
    assert out.all(axis=1)[1] # 2-sigma - all-or-none
    assert (~out).all(axis=1)[2] # 2-sigma - all-or-none
    assert (~out).all(axis=1)[3] # Bad point - all-or-none
