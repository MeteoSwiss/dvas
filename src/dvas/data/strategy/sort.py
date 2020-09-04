"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Sort strategy

"""

# Import from external packages
from abc import ABCMeta, abstractmethod

# Import from current package
from ...database.database import EventManager


class SortDataStrategy(metaclass=ABCMeta):
    """Abstract class to manage data sort strategy"""

    @abstractmethod
    def sort(self, *args, **kwargs):
        """Strategy required method"""


class TimeSortDataStrategy(SortDataStrategy):
    """Class to manage sort of time data parameter"""

    def sort(self, data):
        """Implementation of sort method

        Args:
            data (dict): Dict of list of ProfileManager

        """

        for key, val in data.items():

            # Get index sort position
            _, sort_idx = EventManager.sort(
                [arg.event_mngr for arg in val]
            )

            data.update(
                {key: [val[i] for i in sort_idx]}
            )

        return data
