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


class SaveDataStrategyAbstract(metaclass=ABCMeta):
    """Abstract class to manage data saving strategy"""

    @abstractmethod
    def save(self, *args, **kwargs):
        """Strategy required method"""


class SaveDataStrategy(SaveDataStrategyAbstract):
    """Class to manage saving of time data"""

    def save(self, data, prms):
        """ Implementation of save method.

        Args:
            data (MultiProfile): the data to save into the database.
            prms (list of str): list of column names to save to the
                database.

        """

        # Init
        local_db_linker = LocalDBLinker()

        # Loop on profile
        for prf in data.profiles:

            # Reset index (necessary to fix DataFrame state)
            val = prf.reset_data_index(prf.data)

            # Save to db
            local_db_linker.save(
                [
                    {
                        'index': val.index.values.astype(int),
                        'value': val[prm].values.astype(float),
                        'info': prf.info,
                        'prm_abbr': data.db_variables[prm],
                        'source_info': 'user_script',
                        'force_write': True
                    } for prm in prms if data.db_variables[prm] is not None
                ]
            )
