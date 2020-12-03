"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.database.database module.

"""

# Import external packages and modules
from datetime import datetime, timedelta
import pytz
from copy import deepcopy

# Import from python packages and modules
from dvas.database.database import InfoManager
from dvas.database.database import SearchInfoExpr
from dvas.dvas_environ import glob_var


class TestInfoManager:
    """Test class for InfoManager"""

    dt_test = '20200101T0000Z'
    sn_test = ['1', '0']
    glob_var.evt_id_pat = r'e\:\d'
    evt_tag = 'e:1'
    glob_var.rig_id_pat = r'r\:\d'
    rig_tag = 'r:1'
    glob_var.mdl_id_pat = r'mdl\:\d'
    mdl_tag = 'mdl:1'
    info_mngr = InfoManager(
        dt_test, sn_test, [evt_tag, rig_tag, mdl_tag]
    )

    def test_srn(self):
        """Test getting 'srn' attribute"""
        self.info_mngr.srn = self.sn_test
        assert self.info_mngr.srn == sorted(self.sn_test)

    def test_evt_dt(self):
        """Test setting/getting 'evt_dt' attribute"""
        self.info_mngr.evt_dt = self.dt_test
        assert self.info_mngr.evt_dt == datetime(2020, 1, 1, tzinfo=pytz.UTC)

    def test_evt_id(self):
        """Test getting 'evt_id' attribute"""
        assert self.info_mngr.evt_id == self.evt_tag

    def test_rig_id(self):
        """Test getting 'rig_id' attribute"""
        assert self.info_mngr.rig_id == self.rig_tag

    def test_mdl_id(self):
        """Test getting 'mdl_id' attribute"""
        assert self.info_mngr.mdl_id == self.mdl_tag

    def test_add_tag(self):
        """Test add tag"""

        # Init
        tags_old = self.info_mngr.tags
        tst_tag = 'dummy'

        # Test add
        self.info_mngr.add_tag([tst_tag])
        assert self.info_mngr.tags == sorted(tags_old + [tst_tag])

        # Test double add
        self.info_mngr.add_tag([tst_tag])
        assert self.info_mngr.tags == sorted(tags_old + [tst_tag])

        # Reset
        self.info_mngr.tags = tags_old

    def test_rm_tag(self):
        """Test remove tag"""

        # Init
        tags_old = self.info_mngr.tags
        tst_tag = tags_old[0]

        # Test remove
        self.info_mngr.rm_tag([tst_tag])
        assert self.info_mngr.tags == sorted(tags_old[1:])

        # Reset
        self.info_mngr.tags = tags_old

    def test_sort(self):
        """Test sort method"""

        # Init
        info_mngr_1 = deepcopy(self.info_mngr)
        info_mngr_2 = deepcopy(self.info_mngr)
        info_mngr_2.evt_dt += timedelta(1)

        # Test
        assert all(
            [arg[0] == arg[1] for arg in
            zip(InfoManager.sort([info_mngr_2, info_mngr_1])[0], [info_mngr_1, info_mngr_2])]
        )


def test_search_event_expr_eval():
    """Test SearchInfoExpr.eval static function"""

    # Define
    args = ('trepros1', True)

    # Test all
    assert len(SearchInfoExpr.eval('all()', *args)) > 0

    # Test datetime
    assert (
        SearchInfoExpr.eval(
            'datetime("20180110T0000Z", "==")', *args
        ) ==
        SearchInfoExpr.eval(
            'datetime("20180110T0000Z")', *args
        ) ==
        SearchInfoExpr.eval(
            'dt("20180110T0000Z")', *args
        )
    )

    # Test not_ and or_
    assert (
        SearchInfoExpr.eval(
            'datetime("20180110T0000Z", "==")', *args
        ) ==
        SearchInfoExpr.eval(
            'not_(or_(datetime("20180110T0000Z", "<"), datetime("20180110T0000Z", ">")))',
            *args
        )
    )

    # Test tag
    assert (
        SearchInfoExpr.eval(
            'tag(("e1", "r1"))', *args
        ) ==
        SearchInfoExpr.eval(
            'or_(tag("e1"), tag("r1"))',
            *args
        )
    )

    # Test and_
    assert (
        SearchInfoExpr.eval(
            'and_(tag("e1"), not_(tag("e1")))', *args
        ) ==
        SearchInfoExpr.eval(
            'not_(all())',
            *args
        )
    )
