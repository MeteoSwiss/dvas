"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.data module.

"""

import pytest

# Import from python packages and modules
from dvas.data.io import update_db
from dvas.data.strategy.load import LoadRSProfileStrategy
from dvas.data.data import MultiRSProfile
from dvas.config.definitions.tag import TAG_DERIVED_VAL, TAG_RAW_VAL
from dvas.errors import dvasError

class TestMutliProfile:
    """Test MultiProfile class"""

    # Define
    mlt_prf = MultiRSProfile()

    # Load from alternative method
    update_db('trepros1', strict=True)
    update_db('altpros1', strict=True)
    update_db('tdtpros1', strict=True)

    # Put a MultiRSProfile together
    prf_stgy = LoadRSProfileStrategy()
    data = prf_stgy.load('all()', 'trepros1', 'tdtpros1', alt_abbr='altpros1')
    mlt_prf.update(data[1], data[0])

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

    def test_getter(self):
        """Test getter"""

        # Test profile data
        assert all([
            self.mlt_prf.profiles[i].data.equals(self.data[0][i].data)
            for i in range(len(self.mlt_prf))
        ])

        # Test db_variables
        assert self.mlt_prf.db_variables == self.data[1]

        # Test info
        assert all([
            self.mlt_prf.info[i] == self.data[0][i].info
            for i in range(len(self.mlt_prf))
        ])

    def test_get_prms(self):
        """ Test convenience getter function """

        # Try to get everything out
        out = self.mlt_prf.get_prms(prm_list=None)
        assert all([set(item.columns) == set(self.mlt_prf.profiles[ind].data.columns)
                    for (ind, item) in enumerate(out)])

        # Try to get an index out.
        with pytest.raises(dvasError):
            self.mlt_prf.get_prms(prm_list='alt')


    def test_rm_info_tag(self):
        """Test rm_info_tag method"""

        # Init
        tag_nm = 'test_rm'
        for prf in self.mlt_prf.profiles:
            prf.info.add_tag(tag_nm)
        mlt_prf_1 = self.mlt_prf.copy()

        # Remove
        res = mlt_prf_1.rm_info_tag(tag_nm)
        mlt_prf_2 = self.mlt_prf.rm_info_tag(tag_nm, inplace=False)

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

    def test_add_info_tag(self):
        """Test add_info_tag method"""

        # Init
        tag_nm = 'test_rm'
        for prf in self.mlt_prf.profiles:
            prf.info.rm_tag(tag_nm)
        mlt_prf_1 = self.mlt_prf.copy()

        # Add
        res = mlt_prf_1.add_info_tag(tag_nm)
        mlt_prf_2 = self.mlt_prf.add_info_tag(tag_nm, inplace=False)

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

    def copy(self):
        """Test copy method"""

        # Define
        mlt_prf_copy = self.mlt_prf.copy()

        # Test
        self.mlt_prf_eq(mlt_prf_copy, self.mlt_prf)

    def test_load_from_db(self):
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
        self.mlt_prf_eq(mlt_prf_1, self.mlt_prf)
        self.mlt_prf_eq(mlt_prf_1, mlt_prf_2)

    def test_sort(self):
        """Test sort method"""

        # Define
        mlt_prf_1 = self.mlt_prf.copy()

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

    def test_save_to_db(self):
        """Test save_to_db method"""

        # Define
        tag_nm = 'save_multiprofile'
        mlt_prf_1 = MultiRSProfile()

        # Load from db
        self.mlt_prf.save_to_db(add_tags=[tag_nm])

        # Load from db
        mlt_prf_1.load_from_db(f"tag('{tag_nm}')", 'trepros1', 'tdtpros1', alt_abbr='altpros1')

        # Add tag
        self.mlt_prf.add_info_tag([tag_nm, TAG_DERIVED_VAL])
        self.mlt_prf.rm_info_tag([TAG_RAW_VAL])

        # Compare
        self.mlt_prf_eq(mlt_prf_1, self.mlt_prf)

    def test_append(self):
        """Test append method"""

        # Define
        mlt_prf = MultiRSProfile()

        # Append
        for prf in self.mlt_prf.profiles:
            mlt_prf.append(self.mlt_prf.db_variables, prf)

        # Compare
        self.mlt_prf_eq(mlt_prf, self.mlt_prf)
