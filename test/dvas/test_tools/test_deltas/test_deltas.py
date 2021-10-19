# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing DeltaProfile classes-related tools.

"""

# Import from python packages and modules
import numpy as np
import pytest
import pandas as pd

# Import from this module
from dvas.hardcoded import PRF_REF_TDT_NAME
from dvas.data.strategy.data import CWSProfile, RSProfile, Profile
from dvas.database.database import InfoManager
from dvas.data.data import MultiCWSProfile, MultiRSProfile

# Functions to test
from dvas.tools.deltas.deltas import single_delta, get_deltas

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
def prf_and_cws(db_init):
    """ Return a Profile and a CWSProfile. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_prf = InfoManager('20210302T0000Z', oids[0], tags=['sync', 'e:1', 'r:1'])
    data_prf = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10., 20., 30.], 'flg': [1, 1, 1]})

    # Prepare some datasets to play with
    info_cws = InfoManager('20210302T0000Z', oids[-2:], tags=['cws', 'e:1', 'r:1'])
    data_cws = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10., 20., 30.], 'flg': [1, 1, 1],
                           'tdt': [0e9, 1e9, 2e9], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})

    return Profile(info_prf, data_prf), CWSProfile(info_cws, data_cws)

@pytest.fixture
def rsprf_and_cws(db_init):
    """ Return an RSProfile and a CWSProfile. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_prf = InfoManager('20210302T0000Z', oids[0], tags=['sync', 'e:1', 'r:1'])
    data_prf = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10., 20., 30.], 'flg': [1, 1, 1],
                             'tdt': [0e9, 1e9, 2e9]})

    # Prepare some datasets to play with
    info_cws = InfoManager('20210302T0000Z', oids[-2:], tags=['cws', 'e:1', 'r:1'])
    data_cws = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10., 20., 30.], 'flg': [1, 1, 1],
                           'tdt': [0e9, 1e9, 2e9], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})

    return RSProfile(info_prf, data_prf), CWSProfile(info_cws, data_cws)

@pytest.fixture
def prfs(db_init):
    """ Return a MultiRSProfile. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_1 = InfoManager('20210302T0000Z', oids[0], tags=['e:1', 'r:1'])
    data_1 = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10., 20., 30.], 'flg': [1, 1, 1],
                           'tdt': [1e9, 2e9, 3e9]})
    info_2 = InfoManager('20210302T0000Z', oids[1], tags=['e:1', 'r:1'])
    data_2 = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10.5, 21., np.nan], 'flg': [1, 1, 1],
                           'tdt': [1e9, 2e9, 3e9]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiRSProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None},
                    [RSProfile(info_1, data_1), RSProfile(info_2, data_2)])

    return multiprf

@pytest.fixture
def cwss1(db_init):
    """ Return a MultiRSProfile. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_3 = InfoManager('20210302T0000Z', oids[2], tags=['e:1', 'r:1'])
    data_3 = pd.DataFrame({'alt': [10., 15., 20.], 'val': [11., 21.1, np.nan], 'flg': [1, 1, 1],
                           'tdt': [1e9, 2e9, 3e9], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiCWSProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [CWSProfile(info_3, data_3)])

    return multiprf

@pytest.fixture
def cwss2(db_init):
    """ Return a MultiRSProfile. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_1 = InfoManager('20210302T0000Z', oids[0], tags=['e:1', 'r:1'])
    data_1 = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10., 20., 30.], 'flg': [1, 1, 1],
                           'tdt': [1e9, 2e9, 3e9], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})
    info_2 = InfoManager('20210302T0000Z', oids[1], tags=['e:1', 'r:1'])
    data_2 = pd.DataFrame({'alt': [10., 15., 20.], 'val': [0, 20., np.nan], 'flg': [1, 1, 1],
                           'tdt': [1e9, 2e9, 3e9], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiCWSProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [CWSProfile(info_1, data_1), CWSProfile(info_2, data_2)])

    return multiprf


# Let us test a series of conditions for the different types of uncertainty types
def test_single_delta(prf_and_cws, rsprf_and_cws):
    """ Function used to test if the routine assembling the delta between a GDP and a Profile is ok.

    The function tests:
        - correct assignation of new values, correct tagging, etc ...

    """

    # First check with RSPRofiles
    prf, cws = prf_and_cws
    out = single_delta(prf, cws)
    # Was the data computed properly ?
    assert np.all(out.val == (prf.val - cws.data.droplevel(PRF_REF_TDT_NAME).val))

    # Idem for basic Profiles
    prf, cws = rsprf_and_cws
    out = single_delta(prf, cws)
    # Was the data computed properly ?
    assert np.all(out.val == (prf.val - cws.val).droplevel(PRF_REF_TDT_NAME))

def test_get_deltas(prfs, cwss1, cwss2):
    """ Function to test the creation of a MultiDeltaProfile instance from MultiRSProfile and
    MultiGDPProfile instances.

    The function tests:
        - ...
    """

    # As many CWS as Profiles
    out = get_deltas(prfs, cwss2)

    # Correct length ?
    assert len(out) == 2
    # Correct values ?
    assert np.all(out[0].val.values == (prfs[0].val - cwss2[0].data.val).values)
    assert np.isnan(out[1].val.values[-1])

    # A single CWS for many Profiles
    out = get_deltas(prfs, cwss1)
    # Correct length ?
    assert len(out) == 2
    # Correct values ?
    assert np.all(out[0].val.values[:-1] == (prfs[0].val - cwss1[0].data.val).values[:-1])
    assert np.isnan(out[0].val.values[-1])
    assert np.all(out[1].val.values[:-1] == (prfs[1].val - cwss1[0].data.val).values[:-1])
