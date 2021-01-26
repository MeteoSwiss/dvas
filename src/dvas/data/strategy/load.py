"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Loader strategies

"""

# Import from external packages
import pandas as pd

# Import from current package
from .data import MPStrategyAC
from .data import Profile, RSProfile, GDPProfile
from ..linker import LocalDBLinker
from ...database.database import InfoManager
from ...database.model import Data
from ...errors import LoadError


# Define
INDEX_NM = Data.index.name
VALUE_NM = Data.value.name


class LoadStrategyAC(MPStrategyAC):
    """Abstract load strategy class"""

    @staticmethod
    def fetch(search, **kwargs):
        """A base function that fetches data from the database.

        Args:
            search (str): selection criteria
            **kwargs (dict): Key word parameter to extract.

        Returns:
            list of InfoManager: Data related info
            list of pd.DataFrame: Data

        Example:
            ```
            import dvas.data.strategy.load as ld
            t = ld.LoadProfileStrategy()
            t.fetch("dt('20160715T120000Z', '==')", {'alt':'altpros1', 'val':'trepros1'})
            ```

        """

        # Init
        db_linker = LocalDBLinker()

        # Loop through the requested parameters and extract them from the database.
        res = {
            key: db_linker.load(search, val)
            for key, val in kwargs.items() if val
        }

        # Create tuple of unique info
        info, _ = InfoManager.sort(
            set([arg['info'] for val in res.values() for arg in val])
        )

        # Create DataFrame by concatenation and append
        data = [
            pd.concat(
                [pd.Series(arg['value'], index=arg['index'], name=key)
                 for key, val in res.items() for arg in val if arg['info'] == info_arg],
                axis=1, ignore_index=False
            ) for info_arg in info
        ]

        # Test column name uniqueness
        ass_tst = all(
            tst_tmp := [
                len(arg.columns.unique()) == len(arg.columns)
                for arg in data
            ]
        )
        if not ass_tst:
            err_msg = f'Data with associated metadata {info[tst_tmp.index(False)]} have non unique metadata for a same parameter'
            raise LoadError(err_msg)

        # Add missing columns
        for i in range(len(data)):
            for val in kwargs.keys():
                if val not in data[-1].columns:
                    data[i][val] = None

        return info, data


class LoadProfileStrategy(LoadStrategyAC):
    """Base class to manage the data loading strategy of Profile instances."""

    def execute(self, search, val_abbr, alt_abbr, flg_abbr=None):
        """Execute strategy method to fetch data from the databse.

        Args:
            search (str): selection criteria
            val_abbr (str): name of the parameter values to extract
            alt_abbr (str, optional): name of the altitude parameter to extract.
            flg_abbr (str, optional): name of the flag parameter to extract. Defaults to None.

        """

        # Fetch the data from the database
        db_vs_df_keys = {'val': val_abbr, 'alt': alt_abbr, 'flg': flg_abbr}

        # Fetch data
        info, data = self.fetch(search, **db_vs_df_keys)

        # Create profiles
        out = [Profile(arg[0], data=arg[1]) for arg in zip(info, data)]

        return out, db_vs_df_keys


class LoadRSProfileStrategy(LoadProfileStrategy):
    """Child class to manage the data loading strategy of RSProfile instances."""

    def execute(self, search, val_abbr, tdt_abbr, alt_abbr=None, flg_abbr=None):
        """Execute strategy method to fetch data from the databse.

        Args:
            search (str): selection criteria
            val_abbr (str): name of the parameter values to extract.
            tdt_abbr (str): name of the time delta parameter to extract.
            alt_abbr (str, optional): name of the altitude parameter to extract. Dafaults to None.
            flg_abbr (str, optional): name of the flag parameter to extract. Defaults to None.

        """

        # Fetch the data from the database
        db_vs_df_keys = {'val': val_abbr, 'tdt': tdt_abbr, 'alt': alt_abbr, 'flg': flg_abbr}

        # Fetch data
        info, data = self.fetch(search, **db_vs_df_keys)

        # Create profiles
        out = [RSProfile(arg[0], data=arg[1]) for arg in zip(info, data)]

        return out, db_vs_df_keys


class LoadGDPProfileStrategy(LoadProfileStrategy):
    """Child class to manage the data loading strategy of GDPProfile instances."""

    def execute(
        self, search, val_abbr, tdt_abbr, alt_abbr=None,
        ucr_abbr=None, ucs_abbr=None, uct_abbr=None, ucu_abbr=None,
        flg_abbr=None
    ):
        """Execute strategy method to fetch data from the database.

        Args:
            search (str): selection criteria
            val_abbr (str): name of the parameter values to extract
            tdt_abbr (str): name of the time delta parameter to extract.
            alt_abbr (str, optional): name of the altitude parameter to extract. Dafaults to None.
            ucr_abbr (str, optional): name of the true rig un-correlated uncertainty parameter to
               extract. Defaults to None.
            ucs_abbr (str, optional): name of the true spatial-correlated uncertainty parameter to
               extract. Defaults to None.
            uct_abbr (str, optional): name of the true time-correlated uncertainty parameter to
               extract. Defaults to None.
            ucu_abbr (str, optional): name of the true un-correlated uncertainty parameter to
               extract. Defaults to None.
            flg_abbr (str, optional): name of the flag parameter to extract. Default to None.

        """

        # Fetch the data from the database
        db_vs_df_keys = {
            'val': val_abbr, 'tdt': tdt_abbr, 'alt': alt_abbr,
            'ucr': ucr_abbr, 'ucs': ucs_abbr, 'uct': uct_abbr, 'ucu': ucu_abbr,
            'flg': flg_abbr
        }

        # Fetch data
        info, data = self.fetch(search, **db_vs_df_keys)

        # Create profiles
        out = [GDPProfile(arg[0], data=arg[1]) for arg in zip(info, data)]

        return out, db_vs_df_keys
