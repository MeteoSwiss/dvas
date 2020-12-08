"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.load module.

"""

# Import from python packages and modules
import pytest
import numpy as np

# Import from tested package
from  dvas.data.strategy.data import Profile, RSProfile, GDPProfile
from dvas.data.strategy.load import LoadProfileStrategy, LoadRSProfileStrategy
from dvas.data.strategy.load import LoadGDPProfileStrategy
from dvas.data.linker import LocalDBLinker
from dvas.database.database import InfoManager

from dvas.database.database import DatabaseManager
from dvas.database.model import Data as MdlData
from dvas.database.model import Info as MdlInfo
from dvas.database.model import InfosInstruments, Instrument
from dvas.database.model import DataSource


class TestLoadProfileStrategy:
    """Test for LoadProfileStrategy class"""

    # Define
    loader = LoadProfileStrategy()
    n_data = 3
    index = np.arange(n_data)
    values = np.random.rand(n_data)
    flags_val = np.ones(n_data)
    infos = [
        InfoManager('20200101T0000Z', 'YT-100', 'load_profile'),
        InfoManager('20200202T0000Z', 'YT-100', 'load_profile')
    ]
    db_linker = LocalDBLinker()

    def test_load(self):

        # Create db entry
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.values.astype(float),
                    'info': info, 'prm_abbr': prm,
                    'source_info': 'test_add_data', 'force_write': True
                }
                for prm in ['trepros1', 'altpros1'] for info in self.infos
            ]
        )
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.flags_val.astype(float),
                    'info': info, 'prm_abbr': 'flgpros1',
                    'source_info': 'test_add_data', 'force_write': True
                }
                for info in self.infos
            ]
        )

        # Load entry
        filt = f"tag('load_profile')"
        res = self.loader.load(
            filt, 'trepros1', 'altpros1', flg_abbr='flgpros1'
        )

        # Compare
        assert isinstance(res[0], list)
        assert len(res[0]) > 0
        assert all([(type(arg) == Profile) for arg in res[0]])
        assert all([arg.flg.abs().max() == 1 for arg in res[0]])
        assert isinstance(res[1], dict)


class TestLoadRSProfileStrategy:
    """Test for LoadProfileStrategy class"""

    # Define
    loader = LoadRSProfileStrategy()
    n_data = 3
    index = np.arange(n_data)
    values = np.random.rand(n_data)
    time_val = np.arange(n_data)*1e9
    infos = [
        InfoManager('20200101T0000Z', 'YT-100', 'load_rsprofile'),
        InfoManager('20200202T0000Z', 'YT-100', 'load_rsprofile')
    ]
    db_linker = LocalDBLinker()

    def test_load(self):

        # Create db entry
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.values.astype(float),
                    'info': info, 'prm_abbr': prm,
                    'source_info': 'test_add_data', 'force_write': True
                }
                for prm in ['trepros1', 'altpros1'] for info in self.infos
            ]
        )
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.time_val.astype(float),
                    'info': info, 'prm_abbr': 'tdtpros1',
                    'source_info': 'test_add_data', 'force_write': True
                }
                for info in self.infos
            ]
        )

        # Load entry
        filt = f"tag('load_rsprofile')"
        res = self.loader.load(
            filt, 'trepros1', 'tdtpros1', alt_abbr='altpros1'
        )

        # Compare
        assert isinstance(res[0], list)
        assert len(res[0]) > 0
        assert all([(type(arg) == RSProfile) for arg in res[0]])
        assert all([arg.flg.isna().all() for arg in res[0]])
        assert isinstance(res[1], dict)


class TestLoadGPDProfileStrategy:
    """Test for LoadProfileStrategy class"""

    # Define
    loader = LoadGDPProfileStrategy()
    n_data = 3
    index = np.arange(n_data)
    values = np.random.rand(n_data)
    time_val = np.arange(n_data) * 1e9
    infos = [
        InfoManager('20200101T0000Z', 'YT-100', 'load_gdpprofile'),
        InfoManager('20200202T0000Z', 'YT-100', 'load_gdpprofile')
    ]
    db_linker = LocalDBLinker()

    def test_load(self):
        # Create db entry
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.values.astype(float),
                    'info': info, 'prm_abbr': prm,
                    'source_info': 'test_add_data', 'force_write': True
                }
                for prm in ['trepros1', 'altpros1'] for info in self.infos
            ]
        )
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.time_val.astype(float),
                    'info': info, 'prm_abbr': 'tdtpros1',
                    'source_info': 'test_add_data', 'force_write': True
                }
                for info in self.infos
            ]
        )

        # Load entry
        filt = f"tag('load_gdpprofile')"
        res = self.loader.load(
            filt, 'trepros1', 'tdtpros1', alt_abbr='altpros1'
        )

        # Compare
        assert isinstance(res[0], list)
        assert len(res[0]) > 0
        assert all([(type(arg) == GDPProfile) for arg in res[0]])
        assert all([arg.flg.isna().all() for arg in res[0]])
        assert all([arg.ucr.isna().all() for arg in res[0]])
        assert isinstance(res[1], dict)
