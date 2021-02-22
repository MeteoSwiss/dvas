"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.data module.

"""

# Import from python packages and modules
import numpy as np
import pytest

# Import from python packages and modules
from dvas.data.strategy.load import LoadRSProfileStrategy
from dvas.data.data import MultiRSProfile
from dvas.errors import DvasError


@pytest.fixture
def mlt_prf():
    """# Load multiprofile"""
    mlt_prf = MultiRSProfile()
    prf_stgy = LoadRSProfileStrategy()
    data = prf_stgy.execute("all()", 'trepros1', 'tdtpros1', alt_abbr='altpros1')
    mlt_prf.update(data[1], data[0])
    return mlt_prf


# Define db_data
db_data = {
    'sub_dir': 'test_data_tata',
    'data': [
        {
            'index': np.array([0, 1, 2]),
            'value': np.array([1000, 1001, 1002]),
            'prm_name': prm,
            'info': {
                'evt_dt': dt,
                'type_name': 'YT',
                'srn': 'YT-100', 'pid': '0',
                'tags': 'load_multiprofile',
                'metadata': {},
                'src': ''
            },
        } for dt in ['20200101T0000Z', '20200202T0000Z']
        for prm in ['trepros1', 'altpros1', 'flgpros1', 'tdtpros1']
    ]
}


class TestMutliProfile:
    """Test MultiProfile class"""

    @staticmethod
    def mlt_prf_eq(mlt_prf_1, mlt_prf_2):
        """Test if two MultiProfile are equals"""

        # Sort
        mlt_prf_1.sort()
        mlt_prf_2.sort()

        # Test
        assert id(mlt_prf_1) != id(mlt_prf_2)
        assert (n_data := len(mlt_prf_1)) == len(mlt_prf_2)
        assert all([
            mlt_prf_1.profiles[i].data.equals(mlt_prf_2.profiles[i].data)
            for i in range(n_data)
        ])
        assert mlt_prf_1.db_variables == mlt_prf_2.db_variables
        assert all([
            mlt_prf_1.info[i] == mlt_prf_2.info[i]
            for i in range(n_data)
        ])

    def test_get_prms(self, mlt_prf):
        """ Test convenience getter function """

        # Try to get everything out
        out = mlt_prf.get_prms(prm_list=None)
        assert all([set(item.columns) == set(mlt_prf.profiles[ind].data.columns)
                    for (ind, item) in enumerate(out)])

        # Try to get an index out.
        with pytest.raises(DvasError):
            mlt_prf.get_prms(prm_list='alt')

    def test_rm_info_tags(self, mlt_prf):
        """Test rm_info_tags method"""

        # Init
        tag_nm = 'test_rm'
        for prf in mlt_prf.profiles:
            prf.info.add_tags(tag_nm)
        mlt_prf_1 = mlt_prf.copy()

        # Remove
        res = mlt_prf_1.rm_info_tags(tag_nm)
        mlt_prf_2 = mlt_prf.rm_info_tags(tag_nm, inplace=False)

        # Test inplace = True
        assert res is None

        # Test
        assert all([
            tag_nm not in prf.info.tags
            for prf in mlt_prf_1.profiles
        ])
        assert all([
            tag_nm not in prf.info.tags
            for prf in mlt_prf_2.profiles
        ])

    def test_add_info_tags(self, mlt_prf):
        """Test add_info_tags method"""

        # Init
        tag_nm = 'test_add'
        for prf in mlt_prf.profiles:
            prf.info.rm_tags(tag_nm)
        mlt_prf_1 = mlt_prf.copy()

        # Add
        res = mlt_prf_1.add_info_tags(tag_nm)
        mlt_prf_2 = mlt_prf.add_info_tags(tag_nm, inplace=False)

        # Test inplace = True
        assert res is None

        # Test
        assert all([
            tag_nm in prf.info.tags
            for prf in mlt_prf_1.profiles
        ])
        assert all([
            tag_nm in prf.info.tags
            for prf in mlt_prf_2.profiles
        ])

    def test_copy(self, mlt_prf):
        """Test copy method"""

        # Copy
        mlt_prf_copy = mlt_prf.copy()

        # Test
        self.mlt_prf_eq(mlt_prf_copy, mlt_prf)

    def test_load_from_db(self, mlt_prf):
        """Test load_from_db method"""

        # Define
        mlt_prf_1 = MultiRSProfile()

        # Load from db
        res = mlt_prf_1.load_from_db('all()', 'trepros1', 'tdtpros1', alt_abbr='altpros1')
        mlt_prf_2 = MultiRSProfile().load_from_db('all()', 'trepros1', 'tdtpros1',
                                                  alt_abbr='altpros1', inplace=False)

        # Test inplace = True
        assert res is None

        # Compare
        self.mlt_prf_eq(mlt_prf_1, mlt_prf)
        self.mlt_prf_eq(mlt_prf_1, mlt_prf_2)

    def test_sort(self, mlt_prf):
        """Test sort method"""

        # Define
        mlt_prf_1 = mlt_prf.copy()

        # Sort
        mlt_prf_2 = mlt_prf_1.sort(inplace=False)
        res = mlt_prf_1.sort()

        # Test inplace = True
        assert res is None

        # Compare
        self.mlt_prf_eq(mlt_prf_1, mlt_prf_2)
        assert all([
            prf.info <= prf_2.info
            for i, prf in enumerate(mlt_prf_1.profiles) for prf_2 in mlt_prf_1.profiles[i:]
        ])

    def test_save_to_db(self, mlt_prf, db_init):
        """Test save_to_db method"""

        # Define
        tag_nm = 'save_multiprofile'

        # save from db with no specific
        mlt_prf.save_to_db()

        # save from db with specific tag
        mlt_prf.save_to_db(add_tags=[tag_nm])

    def test_append(self, mlt_prf):
        """Test append method"""

        # Define
        mlt_prf_1 = MultiRSProfile()

        # Append
        for prf in mlt_prf.profiles:
            mlt_prf_1.append(mlt_prf.db_variables, prf)

        # Compare
        self.mlt_prf_eq(mlt_prf, mlt_prf_1)
