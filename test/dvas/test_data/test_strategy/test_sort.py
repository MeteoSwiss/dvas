"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.sort module.

"""

# Import from Python

# Import from tested package
from dvas.data.strategy.sort import SortProfileStrategy
from dvas.data.strategy.data import Profile
from dvas.database.database import InfoManager


# Define db_data
db_data = {
    'sub_dir': 'test_strategy_save',
    'data': [
        {
            'mdl_name': 'YT',
            'srn': 'YT-100',
            'pid': '0',
        },
    ]
}

class TestSortProfileStrategy:
    """Test SortProfileStrategy class"""

    def test_sort(self, db_init):
        """Test sort method"""

        data = db_init.data[0]

        sorter = SortProfileStrategy()
        order = [2, 0, 1]
        prf = [
            Profile(InfoManager('20200110T0000Z', data['oid'])),
            Profile(InfoManager('20200101T0000Z', data['oid'])),
            Profile(InfoManager('20200105T0000Z', data['oid']))
        ]

        # Sort
        prf_sorted = sorter.execute(prf)

        # Test sort result
        assert all(
            prf_sorted[order[i]].info == prf[i].info
            for i in range(len(prf_sorted))
        )
