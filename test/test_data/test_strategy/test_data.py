"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.data module.

"""

# Import from python packages and modules
import pytest
import pandas as pd

from dvas.data.strategy.data import Profile
from dvas.dvas_logger import dvasError


class TestProfile:
    """Test class for Flag Manager"""

    ok_data = pd.DataFrame({'alt': [10., 15., 20.], 'val': [1., 2., 3.], 'flg': [0, 0, 0]})
    ko_index_data = pd.DataFrame({'val': [1, 2, 3], 'flg': [0, 0, 0]})
    inst = Profile(1, ok_data)

    def test_data(self):
        """Test data getter and setter method"""

        # Test data
        assert self.inst.val.astype('float').equals(
            self.ok_data['val'].astype('float')
        )
        assert self.inst.flg.astype('Int64').equals(
            self.ok_data['flg'].astype('Int64')
        )

        # Test setup errors when some data is missing.
        with pytest.raises(dvasError):
             self.inst.data = self.ko_index_data

    def test_copy(self):
        """Test copy method"""
        copy_inst = self.inst.copy()
        assert copy_inst.data.equals(self.inst.data)
