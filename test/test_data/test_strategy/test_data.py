"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.data module.

"""

# Import from python packages and modules
import pytest
import pandas as pd
import numpy as np

# Import from tested package
from dvas.data.strategy.data import Profile, RSProfile, GDPProfile
from dvas.errors import ProfileError
from dvas.database.database import InfoManager

class TestProfile:
    """Test Profile class"""

    info = InfoManager('20200101T0000Z', '0')
    ok_data = pd.DataFrame({'alt': [10., 15., 20.], 'val': [1., 2., 3.], 'flg': [0, 0, 0]})
    ko_index_data = ok_data[['val', 'flg']].copy()

    def test_init(self):
        """Test init class"""

        # Init
        Profile(self.info, self.ok_data)

        # Test argument type info manager
        with pytest.raises(ProfileError):
            Profile(1, self.ok_data)

        # Test bad data
        with pytest.raises(ProfileError):
            Profile(self.info, self.ko_index_data)

    def test_getter(self):
        """Test getter method"""

        inst = Profile(self.info, self.ok_data)

        # Test data
        assert inst.data.equals(
            inst.set_data_index(self.ok_data)
        )

        # Test val
        assert inst.val.equals(
            inst.set_data_index(self.ok_data)['val']
        )

        # Test flg
        assert inst.flg.equals(
            inst.set_data_index(self.ok_data)['flg']
        )

        # Test alt
        assert np.array_equal(inst.alt.values, self.ok_data['alt'].values)

    def test_setter(self):
        """Test setter method"""

        # Init
        inst = Profile(self.info)

        # Test data
        inst.data = self.ok_data

        # Test val
        val_new = inst.val * 2
        inst.val = val_new
        assert inst.val.equals(val_new)

        # Test raises
        with pytest.raises(ProfileError):
            val_bad = inst.val
            val_bad.name = 'xxx'
            inst.val = val_bad

        with pytest.raises(ProfileError):
            inst.val = inst.data[['val']]

        with pytest.raises(AttributeError):
            inst.alt = 0

        # Test for #83:
        inst.data = self.ok_data
        # Try to set the data from an existing profile (i.e. with index columns already set.)
        inst_bis = Profile(self.info, data=inst.data)

        assert (inst_bis.data == inst.data).all

    def test_copy(self):
        """Test copy method"""

        # Init
        inst = Profile(self.info, self.ok_data)

        # Test
        copy_inst = inst.copy()
        assert id(copy_inst) != id(inst)
        assert copy_inst.data.equals(inst.data)


class TestRSProfile:
    """Test RSProfile class"""

    info = InfoManager('20200101T0000Z', '0')
    ok_data = pd.DataFrame(
        {'alt': [10., 15., 20.], 'val': [1., 2., 3.], 'flg': [0, 0, 0], 'tdt': [0, 1e9, 2e9]}
    )
    ko_index_data = ok_data[['val', 'alt', 'flg']].copy()

    def test_init(self):
        """Test init class"""

        # Init
        RSProfile(self.info, self.ok_data)

        # Test bad data
        with pytest.raises(ProfileError):
            RSProfile(self.info, self.ko_index_data)

    def test_getter(self):
        """Test getter method"""

        inst = RSProfile(self.info, self.ok_data)

        # Test data
        assert inst.data.equals(
            inst.set_data_index(self.ok_data)
        )

        # Test val
        assert inst.val.equals(
            inst.set_data_index(self.ok_data)['val']
        )

        # Test flg
        assert inst.flg.equals(
            inst.set_data_index(self.ok_data)['flg']
        )

        # Test alt
        assert np.array_equal(inst.alt.values, self.ok_data['alt'].values)

        # Test tdt
        assert np.array_equal(inst.tdt.values, self.ok_data['tdt'].values)


class TestGDPProfile:
    """Test GDPProfile class"""

    info = InfoManager('20200101T0000Z', '0')
    ok_data = pd.DataFrame(
        {
            'alt': [10., 15., 20.], 'val': [1., 2., 3.], 'flg': [0, 0, 0], 'tdt': [0, 1e9, 2e9],
            'ucr': [1, 1, 1], 'ucs': [1, 1, 1], 'uct': [1, 1, 1], 'ucu': [1, 1, 1]}
    )
    ko_index_data = ok_data[['val', 'alt', 'flg', 'tdt']].copy()

    def test_init(self):
        """Test init class"""

        # Init
        GDPProfile(self.info, self.ok_data)

        # Test bad data
        with pytest.raises(ProfileError):
            GDPProfile(self.info, self.ko_index_data)

    def test_getter(self):
        """Test getter method"""

        # Init
        inst = GDPProfile(self.info, self.ok_data)

        # Test data
        assert inst.data.equals(
            inst.set_data_index(self.ok_data)
        )

        # Test val
        assert inst.val.equals(
            inst.set_data_index(self.ok_data)['val']
        )

        # Test flg
        assert inst.flg.equals(
            inst.set_data_index(self.ok_data)['flg']
        )

        # Test uc_tot
        assert np.round(inst.uc_tot.abs().max(), 1) == np.round(np.sqrt(4), 1)
        inst.uc_tot.name = 'uc_tot'

        # Test alt
        assert np.array_equal(inst.alt.values, self.ok_data['alt'].values)

        # Test tdt
        assert np.array_equal(inst.tdt.values, self.ok_data['tdt'].values)
