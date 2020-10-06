"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.data module.

"""

# Import from python packages and modules
from pathlib import Path
import pytest
import pandas as pd

from dvas.data.strategy.data import TimeProfileManager
from dvas.data.strategy.data import IndexError


# Define
current_pkge_path = Path(__file__).parent
ok_fixture_dir = current_pkge_path / 'etc' / 'ok'
ko_fixture_dir = current_pkge_path / 'etc' / 'ko'


class TestTimeProfileManager:
    """Test class for FlagManager"""

    ok_data = pd.DataFrame(
        {'value': [1, 2, 3], 'flag': [0, 0, 0]},
        index=pd.timedelta_range('1s', '3s', freq='s')
    )
    ko_index_data = pd.DataFrame(
        {'value': [1, 2, 3], 'flag': [0, 0, 0]},
    )

    def test_data(self):
        """Test data getter and setter method"""

        inst = TimeProfileManager(1)
        inst.data = self.ok_data

        # Test
        assert inst.data.equals(self.ok_data)

        # Test index error
        with pytest.raises(IndexError):
            inst.data = self.ko_index_data

    def test_value(self):
        """Test value getter and setter method"""

        inst = TimeProfileManager(1)
        test_series = self.ok_data['value'].copy()
        test_series.name = None

        # Test
        inst.value = test_series
        assert inst.value.equals(test_series)

        # Test index error
        with pytest.raises(IndexError):
            inst.value = self.ko_index_data['value']

    def test_flag(self):
        """Test flag getter and setter method"""

        inst = TimeProfileManager(1)
        test_series = self.ok_data['flag'].copy()
        test_series.name = None

        # Test
        inst.flag = test_series
        assert inst.flag.equals(test_series)

        # Test index error
        with pytest.raises(IndexError):
            inst.flag = self.ko_index_data['flag']

