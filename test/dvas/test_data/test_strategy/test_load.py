"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

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


# Define db_data
db_data = {
    'sub_dir': 'test_strategy_load',
    'data': [
        {
            'index': np.array([0, 1, 2]),
            'value': val,
            'prm_name': prm,
            'info': {
                'edt': dt,
                'mdl_name': 'YT',
                'srn': 'YT-100', 'pid': '0',
                'tags': tags,
                'metadata': {},
                'src': ''
            },
        } for val, dt, tags in [
            (np.array([100, 101, 102]), '20200101T0000Z', ['load_profile', 'e:1']),
            (np.array([200, 201, 202]), '20200202T0000Z', ['load_profile', 'e:2'])
        ] for prm in ['temp', 'gph', 'temp_flag', 'time']
    ]
}


class TestLoadProfileStrategy:
    """Test for LoadProfileStrategy class"""

    def test_load(self):
        """Test load method"""

        # Define
        loader_stgy = LoadProfileStrategy()

        # Load entry
        filt = "tags('load_profile')"
        res = loader_stgy.execute(filt, 'temp', 'gph')

        # Compare
        assert isinstance(res[0], list)
        assert len(res[0]) > 0
        assert all([(type(arg) == Profile) for arg in res[0]])
        assert all([~arg.flg.isna().all() for arg in res[0]])
        assert isinstance(res[1], dict)


class TestLoadRSProfileStrategy:
    """Test for LoadProfileStrategy class"""

    def test_load(self):
        """Test load method"""

        # Define
        loader_stgy = LoadRSProfileStrategy()

        # Load entry
        filt = f"tags('load_profile')"
        res = loader_stgy.execute(
            filt, 'temp', 'time', alt_abbr='gph'
        )

        # Compare
        assert isinstance(res[0], list)
        assert len(res[0]) > 0
        assert all([(type(arg) == RSProfile) for arg in res[0]])
        assert all([~arg.flg.isna().all() for arg in res[0]])
        assert isinstance(res[1], dict)


class TestLoadGDPProfileStrategy:
    """Test for LoadProfileStrategy class"""

    def test_load(self):
        """Test load method"""

        # Define
        loader_stgy = LoadGDPProfileStrategy()

        # Load entry
        filt = f"tags('load_profile')"
        res = loader_stgy.execute(
            filt, 'temp', 'time', alt_abbr='gph'
        )

        # Compare
        assert isinstance(res[0], list)
        assert len(res[0]) > 0
        assert all([(type(arg) == GDPProfile) for arg in res[0]])
        assert all([~arg.flg.isna().all() for arg in res[0]])
        assert all([arg.ucr.isna().all() for arg in res[0]])
        assert isinstance(res[1], dict)
