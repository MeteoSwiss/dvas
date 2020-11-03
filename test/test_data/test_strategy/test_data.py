"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.data module.

"""

# Import from python packages and modules
import pytest
import pandas as pd

from dvas.data.strategy.data import TimeProfileManager


class TestTimeProfileManager:
    """Test class for FlagManager"""

    ok_data = pd.DataFrame(
        {'val': [1, 2, 3], 'flg': [0, 0, 0]},
        index=pd.timedelta_range('1s', '3s', freq='s')
    )
    ko_index_data = pd.DataFrame(
        {'val': [1, 2, 3], 'flg': [0, 0, 0]},
    )
    inst = TimeProfileManager(1, ok_data)

    def test_data(self):
        """Test data getter and setter method"""

        # Test data
        assert self.inst.value.astype('float').equals(
            self.ok_data['val'].astype('float')
        )
        assert self.inst.flag.astype('float').equals(
            self.ok_data['flg'].astype('float')
        )

        # Test index error
        with pytest.raises(IndexError):
            self.inst.data = self.ko_index_data
        with pytest.raises(IndexError):
            self.inst.value = self.ko_index_data['val']
        with pytest.raises(IndexError):
            self.inst.flag = self.ko_index_data['flg']

    def test_copy(self):
        """Test copy method"""
        copy_inst = self.inst.copy()
        assert copy_inst.data.equals(self.inst.data)


