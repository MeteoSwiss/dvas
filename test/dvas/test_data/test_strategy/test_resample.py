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
        data_1 = pd.DataFrame({'alt': [10., 15., 20., 35], 'val': [11., 12., 13., 14],
                               'flg': [0]*4, 'tdt': [0, 1, 1.5, 2.1]})

        # Let's build a multiprofile so I can test things out.
        multiprf = MultiRSProfile()
        multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None},
                        [RSProfile(info_1, data_1)])

        # Let's launch the resampling
        out = multiprf.resample(freq='1s', inplace=False)

        # Can I interpolate properly ?
        assert np.array_equal(out.profiles[0].data.index.get_level_values('alt').values,
                              np.array([10, 15, 32.5]))
        assert np.array_equal(out.profiles[0].data.loc[:, 'val'].values,
                              np.array([11, 12, 14*0.5/0.6+13*(1-0.5/0.6)]))

        # Do I really have 1sec time stamps ?
        tdts = out.profiles[0].data.index.get_level_values('tdt').values.astype('timedelta64[s]')
        assert len(np.unique(np.diff(tdts))) == 1
        assert np.unique(np.diff(tdts))[0] == np.timedelta64(1, 's')

        # Was the flag applied correctly ?
        assert all(out.profiles[0].has_flg('interp') == [False, False, True])

    def test_resample_gdp(self):
        """Test rebase method"""

        # Prepare some datasets to play with
        info_1 = InfoManager('20201217T0000Z', 1)
        data_1 = pd.DataFrame({'alt': [10., 15., 20., 35], 'val': [11., 12., 13., 14],
                               'flg': [0]*4, 'ucr': [1, 1, 1, 1], 'ucs': [1, 1, 1, 1],
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
        assert all(out.profiles[0].has_flg('interp') == [False, False, True])
