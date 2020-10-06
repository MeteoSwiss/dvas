"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Strategy used to save data

"""

# Import from external packages
from abc import ABCMeta, abstractmethod

# Import from current package
from ..linker import LocalDBLinker


class SaveDataStrategy(metaclass=ABCMeta):
    """Abstract class to manage data saving strategy"""

    def __init__(self):
        self._local_db_linker = LocalDBLinker()

    @abstractmethod
    def save(self, *args, **kwargs):
        """Strategy required method"""


class SaveTimeDataStrategy(SaveDataStrategy):
    """Class to manage saving of time data"""

    def save(self, values, event_mngrs, prm_abbr):
        """Implementation of save method

        Args:


        """

        # Convert time index to seconds values
        for key, data_list in values.items():

            event_list = event_mngrs[key]

            for i, arg in enumerate(zip(data_list, event_list)):
                data_list[i].index = arg[0].index.total_seconds()

            # Save to db
            self._local_db_linker.save(
                [
                    {
                        'data': arg_data,
                        'event': arg_event,
                        'prm_abbr': prm_abbr[key],
                    } for arg_data, arg_event in zip(data_list, event_list)
                ]
            )
