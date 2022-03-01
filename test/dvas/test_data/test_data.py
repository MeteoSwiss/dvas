"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.data module.

"""

# Import from python packages and modules
import numpy as np
import pandas as pd
import pytest

# Import from python packages and modules
from dvas.data.strategy.load import LoadRSProfileStrategy, LoadGDPProfileStrategy
from dvas.data.data import MultiProfile, MultiRSProfile, MultiGDPProfile

@pytest.fixture
def mlt_prf():
    """# Load multiprofile"""
    mlt_prf = MultiRSProfile()
    prf_stgy = LoadRSProfileStrategy()
    data = prf_stgy.execute("all()", 'temp', 'time', alt_abbr='gph')
    mlt_prf.update(data[1], data[0])
    return mlt_prf

@pytest.fixture
def mlt_gdpprf():
    """# Load multiprofile"""
    mlt_gdpprf = MultiGDPProfile()
    prf_stgy = LoadGDPProfileStrategy()
    data = prf_stgy.execute("all()", 'temp', 'time', alt_abbr='gph',
                            ucr_abbr='ucr1', ucs_abbr='ucs1', uct_abbr='uct1', ucu_abbr='ucu1')
    mlt_gdpprf.update(data[1], data[0])
    return mlt_gdpprf

# Define db_data
db_data = {
    # TODO
    #  Change dir name
    'sub_dir': 'test_data_tata',
    'data': [{'index': np.array(range(3*(ind+1))),
              'value': np.array([1000, 1001, 1002] * (ind+1)),
              'prm_name': prm,
              'info': {'edt': dt,
                       'mdl_name': 'YT',
                       'srn': 'YT-100', 'pid': '0',
                       'tags': 'load_multiprofile',
                       'metadata': {},
                       'src': ''},
             } for (ind, dt) in enumerate(['20200101T0000Z', '20200202T0000Z'])
             for prm in ['temp', 'gph', 'temp_flag', 'time', 'ucr1', 'ucs1',
                         'uct1', 'ucu1']
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

    def test_iterable(self, mlt_prf):
        """ Test that the MultiProfile is iterable """

        assert [item for item in mlt_prf] == [mlt_prf.profiles[0], mlt_prf.profiles[1]]

    def test_extract(self, mlt_prf):
        """ Test the extraction method of MultiProfile """

        sub_prf = mlt_prf.extract([1])

        # Did I extract the correct amount of profiles, and of the correct type ?
        assert len(sub_prf) == 1
        assert isinstance(sub_prf, type(mlt_prf))
        # Did I get the correct profile out ?
        assert sub_prf[0].info.oid == mlt_prf[1].info.oid

    def test_get_prms(self, mlt_gdpprf):
        """ Test convenience getter function """

        # Try to get everything out
        out = mlt_gdpprf.get_prms(prm_list=None)
        # Correct format ?
        assert isinstance(out, pd.DataFrame)
        # Correct number of profiles ?
        assert len(mlt_gdpprf) == len(np.unique(out.columns.get_level_values('#')))
        # Correct length
        assert len(out) == max([len(prf.data) for prf in mlt_gdpprf])
        # Correct keys for all profiles
        assert all([set(out[ind].columns) | set(out[ind].index.names) ==
                    set(prf.data.columns) | set(prf.data.index.names)
                    for (ind, prf) in enumerate(mlt_gdpprf)])

        # Try to get an index out.
        out = mlt_gdpprf.get_prms(prm_list='alt')
        assert np.unique(out.columns.get_level_values('prm')) == 'alt'

        # Try to get masked data
        # First, set the mask, so I am sure it is there
        mlt_gdpprf[0].set_flg('user_qc', True, [0])
        out_msk = mlt_gdpprf.get_prms(prm_list='val', mask_flgs='user_qc')
        # Then unset it, to see if I can see the data again ...
        mlt_gdpprf[0].set_flg('user_qc', False, [0])
        out_nomsk = mlt_gdpprf.get_prms(prm_list='val', mask_flgs='user_qc')

        #Check that data was masked as I expected
        assert np.isnan(out_msk[0]['val'][0])
        assert not np.isnan(out_nomsk[0]['val'][0])

        # Now check what happens in case I ask for an index, especially a time delta
        mlt_gdpprf[0].set_flg('user_qc', True, [0])
        out_msk = mlt_gdpprf.get_prms(prm_list='tdt', mask_flgs='user_qc')
        assert pd.isnull(out_msk[0]['tdt'][0])

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

        # Test the new has_tag method
        assert all(mlt_prf_1.has_tag('test_add'))
        assert not any(mlt_prf.has_tag('test_add'))
        # Make sure I always get a list no matter what
        assert isinstance(MultiProfile().has_tag('hurray'), list)

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
        res = mlt_prf_1.load_from_db('all()', 'temp', 'time', alt_abbr='gph')
        mlt_prf_2 = MultiRSProfile().load_from_db('all()', 'temp', 'time',
                                                  alt_abbr='gph', inplace=False)

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
