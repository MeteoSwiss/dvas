"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Strategy used to save data

"""

# Import from current package
from .data import MPStrategyAC
from ..linker import LocalDBLinker
from ...hardcoded import PRF_TDT


class SaveDataStrategy(MPStrategyAC):
    """Class to manage saving strategy of time data"""

    def execute(self, data, prms):
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

            # Save to db everything except 'tdt', that requires special treatment
            local_db_linker.save(
                [{'index': val.index.values.astype(int),
                  'value': val[prm].values.astype(float),
                  'info': prf.info,
                  'prm_name': data.db_variables[prm],
                  'force_write': True}
                 for prm in prms
                 if (data.db_variables[prm] is not None) and (prm is not PRF_TDT)]
            )

            local_db_linker.save(
                [{'index': val.index.values.astype(int),
                  'value': val[prm].dt.total_seconds().values.astype(float),
                  'info': prf.info,
                  'prm_name': data.db_variables[prm],
                  'force_write': True}
                 for prm in prms
                 if (data.db_variables[prm] is not None) and (prm is PRF_TDT)]
            )
