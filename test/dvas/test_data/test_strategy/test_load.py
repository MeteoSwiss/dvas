"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.load module.

"""

# Import from python packages and modules
import numpy as np

# Import from tested package
from  dvas.data.strategy.data import Profile, RSProfile, GDPProfile
from dvas.data.strategy.load import LoadProfileStrategy, LoadRSProfileStrategy
from dvas.data.strategy.load import LoadGDPProfileStrategy
from dvas.data.linker import LocalDBLinker
from dvas.database.database import InfoManager


class TestLoadProfileStrategy:
    """Test for LoadProfileStrategy class"""

    # Define
    loader_stgy = LoadProfileStrategy()
    n_data = 3
    index = np.arange(n_data)
    values = np.array([100, 101, 102])
    flags_val = np.ones(n_data)
    sn = 'YT-100'
    infos = [
        InfoManager.from_dict(
            {
                'evt_dt': '20200101T0000Z',
                'srn': sn, 'pid': '0',
                'tags': 'load_profile',
                'metadata': {}
            }
        ),
        InfoManager.from_dict(
            {
                'evt_dt': '20200202T0000Z',
                'srn': sn, 'pid': '0',
                'tags': 'load_profile',
                'metadata': {}
            }
        ),
    ]
    db_linker = LocalDBLinker()

    def test_load(self):
        """Test load method"""

        # Create db entry
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.values.astype(float),
                    'info': info, 'prm_name': prm,
                    'source_info': 'test_add_data', 'force_write': True
                }
                for prm in ['trepros1', 'altpros1'] for info in self.infos
            ]
        )
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.flags_val.astype(float),
                    'info': info, 'prm_name': 'flgpros1',
                    'source_info': 'test_add_data', 'force_write': True
                }
                for info in self.infos
            ]
        )

        # Load entry
        filt = f"tags('load_profile')"
        res = self.loader_stgy.execute(
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
    loader_stgy = LoadRSProfileStrategy()
    n_data = 3
    index = np.arange(n_data)
    values = np.array([200, 201, 202])
    time_val = np.arange(n_data)*1e9
    sn = 'YT-100'
    infos = [
        InfoManager.from_dict(
            {
                'evt_dt': '20200101T0000Z',
                'srn': sn, 'pid': '0',
                'tags': 'load_rsprofile',
                'metadata': {},
            }
        ),
        InfoManager.from_dict(
            {
                'evt_dt': '20200202T0000Z',
                'srn': sn, 'pid': '0',
                'tags': 'load_rsprofile',
                'metadata': {},
            }
        ),
    ]
    db_linker = LocalDBLinker()

    def test_load(self):
        """Test load method"""

        # Create db entry
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.values.astype(float),
                    'info': info, 'prm_name': prm,
                    'source_info': 'test_add_data', 'force_write': True
                }
                for prm in ['trepros1', 'altpros1'] for info in self.infos
            ]
        )
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.time_val.astype(float),
                    'info': info, 'prm_name': 'tdtpros1',
                    'source_info': 'test_add_data', 'force_write': True
                }
                for info in self.infos
            ]
        )

        # Load entry
        filt = f"tags('load_rsprofile')"
        res = self.loader_stgy.execute(
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
    loader_stgy = LoadGDPProfileStrategy()
    n_data = 3
    index = np.arange(n_data)
    values = np.array([300, 301, 302])
    time_val = np.arange(n_data) * 1e9
    sn = 'YT-100'
    infos = [
        InfoManager.from_dict(
            {
                'evt_dt': '20200101T0000Z',
                'srn': sn, 'pid': '0',
                'tags': 'load_gdpprofile',
                'metadata': {},
            }
        ),
        InfoManager.from_dict(
            {
                'evt_dt': '20200202T0000Z',
                'srn': sn, 'pid': '0',
                'tags': 'load_gdpprofile',
                'metadata': {},
            }
        ),
    ]
    db_linker = LocalDBLinker()

    def test_load(self):
        """Test load method"""

        # Create db entry
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.values.astype(float),
                    'info': info, 'prm_name': prm,
                    'source_info': 'test_add_data', 'force_write': True
                }
                for prm in ['trepros1', 'altpros1'] for info in self.infos
            ]
        )
        self.db_linker.save(
            [
                {
                    'index': self.index.astype(int), 'value': self.time_val.astype(float),
                    'info': info, 'prm_name': 'tdtpros1',
                    'source_info': 'test_add_data', 'force_write': True
                }
                for info in self.infos
            ]
        )

        # Load entry
        filt = f"tags('load_gdpprofile')"
        res = self.loader_stgy.execute(
            filt, 'trepros1', 'tdtpros1', alt_abbr='altpros1'
        )

        # Compare
        assert isinstance(res[0], list)
        assert len(res[0]) > 0
        assert all([(type(arg) == GDPProfile) for arg in res[0]])
        assert all([arg.flg.isna().all() for arg in res[0]])
        assert all([arg.ucr.isna().all() for arg in res[0]])
        assert isinstance(res[1], dict)