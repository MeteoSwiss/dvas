"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.resample submodule.

"""

# Import from python packages and modules
import numpy as np
import pandas as pd

# Import from tested package
from dvas.data.strategy.data import RSProfile, GDPProfile
from dvas.data.data import MultiRSProfile, MultiGDPProfile
from dvas.database.database import InfoManager
from dvas.hardcoded import FLG_INTERP

# Define db_data
db_data = {
    'sub_dir': 'test_strategy_resample',
    'data': [
        {
            'mdl_name': 'YT',
            'srn': 'YT-100',
            'pid': '0',
        },
    ]
}


class TestResampleStrategy:
    """Test SortProfileStrategy class"""

    def test_resample_rs(self):
        """Test resample  method"""

        # Prepare some datasets to play with
        info_1 = InfoManager('20201217T0000Z', 1)
        data_1 = pd.DataFrame({'alt': [10., 15., 20., 35, 35], 'val': [11., 12., 13., 14, 14],
                               'flg': [0]*5, 'tdt': [0, 1, 1.5, 2.1, 5]})

        # Let's build a multiprofile so I can test things out.
        multiprf = MultiRSProfile()
        multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None},
                        [RSProfile(info_1, data_1)])

        # Let's launch the resampling
        out = multiprf.resample(freq='1s', interp_dist=1, inplace=False)

        # Can I interpolate properly ?
        assert np.array_equal(out.profiles[0].data.index.get_level_values('alt').values,
                              np.array([10, 15, 32.5, 35, np.nan, 35]),
                              equal_nan=True)
        assert np.array_equal(out.profiles[0].data.loc[:, 'val'].values,
                              np.array([11, 12, 14*0.5/0.6+13*(1-0.5/0.6), 14, np.nan, 14]),
                              equal_nan=True)

        # Do I really have 1 sec time stamps ?
        tdts = out.profiles[0].data.index.get_level_values('tdt').values.astype('timedelta64[s]')
        assert len(np.unique(np.diff(tdts))) == 1
        assert np.unique(np.diff(tdts))[0] == np.timedelta64(1, 's')

        # Was the flag applied correctly ?
        assert all(out.profiles[0].has_flg(FLG_INTERP) == [False, False, True, True, True, False])

    def test_resample_gdp(self):
        """Test rebase method"""

        # Prepare some datasets to play with
        info_1 = InfoManager('20201217T0000Z', 1)
        data_1 = pd.DataFrame({'alt': [10., 15., 20., 35], 'val': [11., 12., 13., 14],
                               'flg': [0, 1, 2, 8], 'ucr': [1, 1, 1, 1], 'ucs': [1, 1, 1, 1],
                               'uct': [1, 1, 1, 1], 'ucu': [1, 1, 1, 1],
                               'tdt': [0, 1, 1.5, 2.1]})

        # Let's build a multiprofile so I can test things out.
        multiprf = MultiGDPProfile()
        multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None,
                         'ucs': None, 'uct': None, 'ucu': None},
                        [GDPProfile(info_1, data_1)])

        # Let's launch the resampling
        out = multiprf.resample(freq='1s', inplace=False)

        # The weight factor for the last item. To save me typing it everywhere
        w = 0.5/0.6

        # Proper interpolation
        assert all(out.profiles[0].data.loc[2, 'val'] ==
                   data_1.loc[2, 'val']*(1-w) + data_1.loc[3, 'val']*w)
        assert out.profiles[0].data.index.get_level_values('alt')[2] == \
            data_1.loc[2, 'alt']*(1-w) + data_1.loc[3, 'alt']*w

        # Proper error propagation
        assert all(out.profiles[0].data.loc[2, 'ucu'] == np.sqrt((1-w)**2 + w**2))
        assert all(out.profiles[0].data.loc[2, 'ucr'] == np.sqrt((1-w)**2 + w**2))
        assert all(out.profiles[0].data.loc[2, 'ucs'] == 1)
        assert all(out.profiles[0].data.loc[2, 'uct'] == 1)

        # Was the flag applied correctly ?
        assert all(out.profiles[0].has_flg(FLG_INTERP) == [False, False, True])
        assert np.array_equal(out[0].flg.values, [0, 1, 14])  # FLG_INTERP is bit 8

