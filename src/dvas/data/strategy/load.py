"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Loader strategies

"""

# Import from external packages
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np

# Import from current package
from .data import TimeProfileManager
from ..linker import LocalDBLinker
from ...database.model import Data


# Define
INDEX_NM = Data.index.name
VALUE_NM = Data.value.name


class LoadDataStrategy(metaclass=ABCMeta):
    """Abstract class to manage data loading strategy"""

    def __init__(self):
        self._db_linker = LocalDBLinker()

    @abstractmethod
    def load(self, *args, **kwargs):
        """Strategy required method"""


class LoadTimeDataStrategy(LoadDataStrategy):
    """Class to manage loading of same time data parameter"""

    def load(self, search, prm_abbr):
        """Implementation of load method"""

        # Get data
        res = self._db_linker.load(search, prm_abbr)

        # Test
        assert len(res) > 0, "No data"

        # Format dataframe index
        for i, arg in enumerate(res):
            res[i]['data'][INDEX_NM] = pd.TimedeltaIndex(arg['data'][INDEX_NM], 's')
            res[i]['data'] = arg['data'].set_index([INDEX_NM])[[VALUE_NM]]

            #TODO
            # Load flag from DB too
            res[i]['data']['flag'] = np.nan

        # Load data
        out = [TimeProfileManager(data['event'], data=data['data']) for data in res]

        return {'data': out}


class LoadAltDataStrategy(LoadTimeDataStrategy):
    """Class to manage loading of same time data parameter"""

    def load(self, search, prm_abbr, alt_prm_abbr):
        """Implementation of load method"""

        # Load prm_abbr
        out_prm = super().load(search, prm_abbr)

        # Load alt
        out_alt_prm = super().load(search, alt_prm_abbr)

        return {'data': out_prm['data'], 'alt': out_alt_prm['data']}
