"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.database.database module.

"""

# Import external packages and modules
import re
from datetime import datetime, timedelta
import pytz
from copy import deepcopy
import pytest
import numpy as np

# Import from python packages and modules
from dvas.database.model import Parameter as MdlParameter
from dvas.database.model import Instrument as MdlInstrument
from dvas.database.model import InstrType as MdlInstrType
from dvas.database.database import DatabaseManager
from dvas.database.database import InfoManager
from dvas.database.database import SearchInfoExpr
from dvas.database.database import OneDimArrayConfigLinker
from dvas.database.database import ConfigGenMaxLenError
from dvas.database.database import DBInsertError
from dvas.dvas_environ import glob_var, path_var
from dvas.config.config import Parameter


@pytest.fixture
def db_mngr():
    return DatabaseManager()


class TestDatabaseManager:
    """Test DatabaseManager class"""

    # Define
    n_data = 3
    index = np.arange(n_data)
    values = np.random.rand(n_data)
    prm = 'trepros1'
    info = InfoManager('20200101T0000Z', 'YT-100')

    def test_get_or_none(self, db_mngr):
        """Test get_or_none method"""

        assert db_mngr.get_or_none(
            MdlParameter,
            search={
                'where': MdlParameter.prm_abbr.in_(
                    ['dummytst_param1', 'dummytst_param2']
                )
            },
            attr=[[MdlParameter.prm_abbr.name]],
        ) == ['dummytst_param1']

        assert db_mngr.get_or_none(
            MdlInstrument,
            search={
                'where': MdlInstrType.type_name == 'YT',
                'join_order': [MdlInstrType],
            },
            attr=[[MdlInstrument.srn.name]],
            get_first=True
        ) in [['YT-100']]

        assert db_mngr.get_or_none(
            MdlParameter,
            search={
                'where': MdlParameter.prm_abbr.in_(
                    ['dummytst_param1', 'dummytst_param2']
                )
            },
            attr=[[MdlParameter.prm_abbr.name]],
            get_first=False
        ) == [['dummytst_param1'], ['dummytst_param2']]

    def test_add_data(self, db_mngr):
        """Test add_data method"""

        # Test add data
        db_mngr.add_data(
            self.index,
            self.values,
            self.info,
            self.prm, source_info='test_add_data'
        )

        with pytest.raises(AssertionError):
            db_mngr.add_data(
                [],
                self.values,
                self.info,
                self.prm, source_info='test_add_data'
            )

        with pytest.raises(AssertionError):
            db_mngr.add_data(
                self.index,
                [],
                self.info,
                self.prm, source_info='test_add_data'
            )

        with pytest.raises(DBInsertError):
            db_mngr.add_data(
                self.index,
                self.values,
                self.info,
                'xxxxxxx', source_info='test_add_data'
            )

    def test_get_data(self, db_mngr):
        """Test get_data method"""

        db_mngr.get_data(
            f"and(dt('{self.info.evt_dt}'), sn('{self.info.srn}'))",
            'trepros1', True
        )

    def test_get_flags(self, db_mngr):
        """Test get_flags"""
        assert isinstance(db_mngr.get_flags(), list)


class TestOneDimArrayConfigLinker:
    """Test OneDimArrayConfigLinker class

    Tests:
        - String generator
        - Catch generated string

    """

    # String generator patern
    prm_pat = re.compile(r'^dummytst_(param\d+)$')

    # Catched string patern
    desc_pat = re.compile(r'^param\d+$')

    # Config linker
    cfg_lkr = OneDimArrayConfigLinker([Parameter])

    def test_get_document(self):
        """Test get_document method

        Test:
            Returned type
            Item generator
            Raises ConfigGenMaxLenError

        """

        doc = self.cfg_lkr.get_document(Parameter.CLASS_KEY)

        assert isinstance(doc, list)
        assert sum(
            [self.prm_pat.match(arg['prm_abbr']) is not None for arg in doc]
        ) == 10
        assert sum(
            [self.desc_pat.match(arg['prm_desc']) is not None for arg in doc]
        ) == 10
        with glob_var.set_many_attr({'config_gen_max': 2}):
            with pytest.raises(ConfigGenMaxLenError):
                self.cfg_lkr.get_document(Parameter.CLASS_KEY)


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
