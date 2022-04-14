"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Sort strategy

"""

# Import from current package
from .data import MPStrategyAC
from ...database.database import InfoManager


class SortProfileStrategy(MPStrategyAC):
    """Class to manage sort of time data parameter"""

    def execute(self, data):
        """Implementation of sort method

        Args:
            data (list): list of Profile

        Returns:
            list of Profile

        """

        # Get index sort position
        _, sort_idx = InfoManager.sort(
            [arg.info for arg in data]
        )

        # Arrange position
        data = [data[i] for i in sort_idx]

        return data
