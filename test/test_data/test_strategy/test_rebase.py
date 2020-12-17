"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.rebase submodule.

"""

import numpy as np
import pandas as pd

# Import from tested package
from dvas.data.strategy.rebase import RebaseStrategy
from dvas.data.strategy.data import RSProfile
from dvas.data.data import MultiRSProfile
from dvas.database.database import InfoManager


class TestRebaseStrategy:
    """Test SortProfileStrategy class"""

    # Prepare some datasets to play with
    info_1 = InfoManager('20201217T0000Z', '0')
    data_1 = pd.DataFrame({'alt': [10., 15., 20.], 'val': [11., 21., 31.], 'flg': [1, 1, 1],
                           'tdt': [10, 20, 30]})
    data_2 = pd.DataFrame({'alt': [1., 1., 2., 5.], 'val': [1., 2., 3., 4.], 'flg': [0, 0, 0, 0],
                           'tdt': [0, 1, 2, 3]})

    def test_rebase(self):
        """Test rebase method"""

        # Let's build a multiprofile so I can test things out.
        multiprf = MultiRSProfile()
        multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None},
                        [RSProfile(self.info_1, self.data_1),
                         RSProfile(self.info_1, self.data_2)])

        # Cropping, no shift
        # Let's set all the Profiles on the same index set as the first
        ref_ind = multiprf.profiles[0].data.index
        out_a = multiprf.rebase(ref_ind, shift=0, inplace=False)
        assert len(out_a.profiles[1].data) == len(self.data_1) # All has the proper length
        assert all(out_a.profiles[1].data.index == multiprf.profiles[0].data.index) # Index are ok
        assert np.all(out_a.profiles[1].data.to_numpy() ==
                      multiprf.profiles[1].data[:-1].to_numpy()) # Data was cropped as requested.

        # Gap filling, no shift
        # Let's set all the Profiles on the same index set as the second
        ref_ind = multiprf.profiles[1].data.index
        out_b = multiprf.rebase(ref_ind, shift=0, inplace=False)
        assert len(out_b.profiles[0].data) == len(self.data_2) # All has the proper length.
        assert np.all(out_b.profiles[1].data.to_numpy() ==
                      multiprf.profiles[1].data.to_numpy()) # Data has not changed.
        assert out_b.profiles[0].data[-1:].isna().all(axis=None) # NaN's where added as needed.

        # Shift things around a bit
        ref_ind = multiprf.profiles[1].data.index
        out_c = multiprf.rebase(ref_ind, shift=1, inplace=False)
        assert out_c.profiles[0].data[:1].isna().all(axis=None) # NaN's where added as needed.
        assert out_c.profiles[1].data[:1].isna().all(axis=None) # NaN's where added as needed.
        assert np.all(out_c.profiles[0].data[1:].to_numpy() ==
                      multiprf.profiles[0].data.to_numpy()) # Data has moved where it should
        assert np.all(out_c.profiles[1].data[1:].to_numpy() ==
                      multiprf.profiles[1].data[:-1].to_numpy()) # Data has moved where it should

        # Shift things around a bit ... but in the other direction
        ref_ind = multiprf.profiles[1].data.index
        out_d = multiprf.rebase(ref_ind, shift=-2, inplace=False)
        assert out_d.profiles[0].data[-2:].isna().all(axis=None) # NaN's where added as needed.
        assert out_d.profiles[1].data[-2:].isna().all(axis=None) # NaN's where added as needed.
        assert np.all(out_d.profiles[0].data[:-3].to_numpy() ==
                      multiprf.profiles[0].data[2:].to_numpy()) # Data has moved where it should
        assert np.all(out_d.profiles[1].data[:-2].to_numpy() ==
                      multiprf.profiles[1].data[2:].to_numpy()) # Data has moved where it should

        # Inplace changes
        ref_ind = multiprf.profiles[1].data.index
        out_e = multiprf.rebase(ref_ind, shift=-2, inplace=True)
        assert out_e is None
        assert len(multiprf.profiles[0].data) == len(self.data_2) # All has the proper length
