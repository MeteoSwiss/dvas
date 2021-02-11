"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.load module.

"""

# Import from python packages and modules
import numpy as np

# Import from tested package
from dvas.data.strategy.data import Profile, RSProfile, GDPProfile
from dvas.data.strategy.load import LoadProfileStrategy, LoadRSProfileStrategy
from dvas.data.strategy.load import LoadGDPProfileStrategy

# Import from current package
from ...db_fixture import db_init  # noqa pylint: disable=W0611


# Define db_data
db_data = {
    'sub_dir': 'test_strategy_load',
    'data': [
        {
            'index': np.array([0, 1, 2]),
            'value': val,
            'prm_name': prm,
            'info': {
                'evt_dt': dt,
                'type_name': 'YT',
                'srn': 'YT-100', 'pid': '0',
                'tags': 'load_profile',
                'metadata': {}
            },
            'source_info': 'test_add_data'
        } for val, dt in [
            (np.array([100, 101, 102]), '20200101T0000Z'),
            (np.array([200, 201, 202]), '20200202T0000Z')
        ] for prm in ['trepros1', 'altpros1', 'flgpros1', 'tdtpros1']
    ]
}


class TestLoadProfileStrategy:
    """Test for LoadProfileStrategy class"""

    def test_load(self, db_init):
        """Test load method"""

        # Define
        loader_stgy = LoadProfileStrategy()

        # Load entry
        filt = f"tags('load_profile')"
        res = loader_stgy.execute(
            filt, 'trepros1', 'altpros1', flg_abbr='flgpros1'
        )

        # Compare
        assert isinstance(res[0], list)
        assert len(res[0]) > 0
        assert all([(type(arg) == Profile) for arg in res[0]])
        assert all([~arg.flg.isna().all() for arg in res[0]])
        assert isinstance(res[1], dict)


class TestLoadRSProfileStrategy:
    """Test for LoadProfileStrategy class"""

    def test_load(self, db_init):
        """Test load method"""

        # Define
        loader_stgy = LoadRSProfileStrategy()

        # Load entry
        filt = f"tags('load_profile')"
        res = loader_stgy.execute(
            filt, 'trepros1', 'tdtpros1', alt_abbr='altpros1'
        )

        # Compare
        assert isinstance(res[0], list)
        assert len(res[0]) > 0
        assert all([(type(arg) == RSProfile) for arg in res[0]])
        assert all([arg.flg.isna().all() for arg in res[0]])
        assert isinstance(res[1], dict)


class TestLoadGDPProfileStrategy:
    """Test for LoadProfileStrategy class"""

    def test_load(self, db_init):
        """Test load method"""

        # Define
        loader_stgy = LoadGDPProfileStrategy()

        # Load entry
        filt = f"tags('load_profile')"
        res = loader_stgy.execute(
            filt, 'trepros1', 'tdtpros1', alt_abbr='altpros1'
        )

        # Compare
        assert isinstance(res[0], list)
        assert len(res[0]) > 0
        assert all([(type(arg) == GDPProfile) for arg in res[0]])
        assert all([arg.flg.isna().all() for arg in res[0]])
        assert all([arg.ucr.isna().all() for arg in res[0]])
        assert isinstance(res[1], dict)
