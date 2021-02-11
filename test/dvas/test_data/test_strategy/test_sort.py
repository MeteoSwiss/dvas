"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.sort module.

"""


# Import from tested package
from dvas.data.strategy.sort import SortProfileStrategy
from dvas.data.strategy.data import Profile
from dvas.database.database import InfoManager


class TestSortProfileStrategy:
    """Test SortProfileStrategy class"""

    sorter = SortProfileStrategy()
    order = [2, 0, 1]
    prf = [
        Profile(InfoManager('20200110T0000Z', [0])),
        Profile(InfoManager('20200101T0000Z', [0])),
        Profile(InfoManager('20200105T0000Z', [0]))
    ]

    def test_sort(self):
        """Test sort method"""

        # Sort
        prf_sorted = self.sorter.execute(self.prf)

        # Test sort result
        assert all(
            prf_sorted[self.order[i]].info == self.prf[i].info
            for i in range(len(prf_sorted))
        )
