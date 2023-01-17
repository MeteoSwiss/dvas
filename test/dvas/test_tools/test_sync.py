"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.tools.sync module.

"""

# Import from Python
import pytest
import numpy as np
import pandas as pd

# Import from dvas
from dvas.data.strategy.data import GDPProfile
from dvas.data.data import MultiGDPProfile
from dvas.database.database import InfoManager

# Functions to test
import dvas.tools.sync as dts

# Define db_data. This is some black magic that is directly related to conftest.py
# This is a temporary db, that is required for get_info() to work properly with mdl_id.
# It all relies on the db config files located in the processing arena
db_data = {
    'sub_dir': 'test_tool_stats',
    'data': [{'mdl_name': 'AR-GDP_001',
              'srn': 'AR1',
              'pid': '0'},
             {'mdl_name': 'BR-GDP_001',
              'srn': 'BR1',
              'pid': '1'},
             ]}


@pytest.fixture
def prfs(db_init):
    """ Return a MultiGDPProfile with 2 GDPprofiles in it. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_1 = InfoManager('20210302T0000Z', oids[0], tags=['e:1', 'r:1'])
    data_1 = pd.DataFrame({'alt': [4995.2, 5000.1, 5005.3, 5010.9],
                           'val': [0, 10., 20., 30.],
                           'flg': [1, 1, 1, 1], 'tdt': [0, 1, 2, 3],
                           'ucs': [1, 1, 1, 1], 'uct': [1, 1, 1, 1], 'ucu': [1, 1, 1, 1]})
    info_2 = InfoManager('20210302T0000Z', oids[1], tags=['e:1', 'r:1'])
    data_2 = pd.DataFrame({'alt': [5001.2, 5003.1, 5011.2, 5015.0],
                           'val': [2, 14., 26., np.nan],
                           'flg': [1, 1, 1, 1], 'tdt': [0, 1, 2, 3],
                           'ucs': [1, 1, 1, 1], 'uct': [1, 1, 1, 1], 'ucu': [1, 1, 1, 1]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [GDPProfile(info_1, data_1), GDPProfile(info_2, data_2)])

    return multiprf


def test_get_sync_shifts_from_alt(prfs):
    """ Test get_sync_shifts_from_alt() """

    out = dts.get_sync_shifts_from_alt(prfs, ref_alt=5000.)

    assert isinstance(out, list)
    assert len(out) == len(prfs)
    assert out == [0, 1]


def test_get_sync_shifts_from_val(prfs):
    """ Test get_sync_shifts_from_val """

    out = dts.get_sync_shifts_from_val(prfs, max_shift=100, first_guess=None)

    assert isinstance(out, list)
    assert len(out) == len(prfs)
    # Since I am doing a relative sync w.r.t the first profile, the first shift should always be 0.
    assert out[0] == 0
