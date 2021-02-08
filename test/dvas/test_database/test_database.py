"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.database.database module.

"""

# Import external packages and modules
from copy import deepcopy
from datetime import datetime, timedelta
import pytz
import pytest
import numpy as np

# Import from current python packages and modules
from dvas.database.model import Parameter as TableParameter
from dvas.database.model import Object as TableObject
from dvas.database.model import InstrType as TableInstrType
from dvas.database.database import DatabaseManager
from dvas.database.database import InfoManager, InfoManagerMetaData
from dvas.database.database import SearchInfoExpr
from dvas.database.database import DBInsertError
from dvas.environ import glob_var


@pytest.fixture
def db_mngr():
    """Get DatabaseManager"""
    return DatabaseManager()


class TestDatabaseManager:
    """Test DatabaseManager class"""

    # Define
    n_data = 3
    index = np.arange(n_data)
    values = np.array([550, 551, 552])
    prm = 'trepros1'
    sn = 'YT-100'
    info = InfoManager.from_dict(
        {
            'evt_dt': '20200101T0000Z',
            'srn': sn, 'pid': '0',
            'tags': 'data_test_db',
            'metadata': {'test_key_str': 'one', 'test_key_num': '1'}
        }
    )

    def test_get_or_none(self, db_mngr):
        """Test get_or_none method"""

        assert db_mngr.get_or_none(
            TableParameter,
            search={
                'where': TableParameter.prm_name.in_(
                    ['dummytst_param1', 'dummytst_param2']
                )
            },
            attr=[[TableParameter.prm_name.name]],
        ) == ['dummytst_param1']

        assert db_mngr.get_or_none(
            TableObject,
            search={
                'where': TableInstrType.type_name == 'YT',
                'join_order': [TableInstrType],
            },
            attr=[[TableObject.srn.name]],
            get_first=True
        ) in [['YT-100'], ['YT-101']]

        assert db_mngr.get_or_none(
            TableParameter,
            search={
                'where': TableParameter.prm_name.in_(
                    ['dummytst_param1', 'dummytst_param2']
                )
            },
            attr=[[TableParameter.prm_name.name]],
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

        # Test add same data (overwrite)
        db_mngr.add_data(
            self.index,
            self.values,
            self.info,
            self.prm, source_info='test_add_data',
            force_write=True
        )

        # Test add same data (no overwrite)
        db_mngr.add_data(
            self.index,
            self.values,
            self.info,
            self.prm, source_info='test_add_data',
            force_write=False
        )

        with pytest.raises(DBInsertError):
            db_mngr.add_data(
                [],
                self.values,
                self.info,
                self.prm, source_info='test_add_data'
            )

        with pytest.raises(DBInsertError):
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

        res = db_mngr.get_data(
            f"and_(dt('{self.info.evt_dt}'), srn('{self.sn}'), tags('data_test_db'))",
            'trepros1', True
        )

        assert isinstance(res, list)
        assert all([isinstance(arg, dict) for arg in res])
        assert all([arg.keys() == set(['info', 'index', 'value']) for arg in res])
        assert all([isinstance(arg['info'], InfoManager) for arg in res])
        assert all([len(arg['index']) == len(arg['value']) for arg in res])
        assert all([len(arg['index']) == self.n_data for arg in res])

    def test_get_flags(self, db_mngr):
        """Test get_flags"""
        assert isinstance(db_mngr.get_flags(), list)


class TestInfoManagerMetaData:
    """Test class for InfoManagerMetaData"""

    inst = InfoManagerMetaData({})

    def test_copy(self):
        """Test copy method"""
        assert id(self.inst.copy()) != id(self.inst)

    def test_update(self):
        """Test update method"""
        # Test int -> float
        self.inst.update({'a': 1})
        assert self.inst['a'] == 1.

        # Test float
        self.inst.update({'a': 1.})
        assert self.inst['a'] == 1.

        # Test str
        self.inst.update({'a': 'one'})
        assert self.inst['a'] == 'one'

        # Test not str, float or int
        with pytest.raises(TypeError):
            self.inst.update({'a': [1]})


class TestInfoManager:
    """Test class for InfoManager"""

    dt_test = '20200101T0000Z'
    oid_test = [1, 2]
    glob_var.evt_id_pat = r'e\:\d'
    evt_tag = 'e:1'
    glob_var.rig_id_pat = r'r\:\d'
    rig_tag = 'r:1'
    glob_var.mdl_id_pat = r'mdl\:\d'
    mdl_tag = 'mdl:1'
    metadata = {'key_str': 'one', 'key_num': 1.}
    info_mngr = InfoManager(
        dt_test, oid_test, tags=[evt_tag, rig_tag, mdl_tag], metadata=metadata
    )

    def test_oid(self):
        """Test getting 'oid' attribute"""
        self.info_mngr.oid = self.oid_test
        assert self.info_mngr.oid == sorted(self.oid_test)

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

    def test_add_tags(self):
        """Test add_tags method"""

        # Init
        tags_old = self.info_mngr.tags
        tst_tag = 'dummy'

        # Test add
        self.info_mngr.add_tags([tst_tag])
        assert self.info_mngr.tags == sorted(tags_old + [tst_tag])

        # Test double add
        self.info_mngr.add_tags([tst_tag])
        assert self.info_mngr.tags == sorted(tags_old + [tst_tag])

        # Reset
        self.info_mngr.tags = tags_old

    def test_rm_tags(self):
        """Test rm_tags method"""

        # Init
        tags_old = self.info_mngr.tags
        tst_tag = tags_old[0]

        # Test remove
        self.info_mngr.rm_tags([tst_tag])
        assert self.info_mngr.tags == sorted(tags_old[1:])

        # Reset
        self.info_mngr.tags = tags_old

    def test_add_metadata(self):
        """Test add_metadata"""

        self.info_mngr.add_metadata('a', 1)
        assert self.info_mngr.metadata['a'] == 1

    def test_rm_metadata(self):
        """Test add_metadata"""

        self.info_mngr.add_metadata('a', 1)
        self.info_mngr.rm_metadata('a')

        with pytest.raises(KeyError):
            self.info_mngr.metadata['a']

    def test_sort(self):
        """Test sort method"""

        # Init
        info_mngr_1 = deepcopy(self.info_mngr)
        info_mngr_2 = deepcopy(self.info_mngr)
        info_mngr_2.evt_dt += timedelta(1)
        info_mngr_3 = deepcopy(info_mngr_2)
        info_mngr_3.oid = [3] + info_mngr_3.oid

        # Test
        assert all(
            [
                arg[0] == arg[1] for arg in
                zip(
                    InfoManager.sort(
                        [info_mngr_3, info_mngr_2, info_mngr_1]
                    )[0],
                    [info_mngr_1, info_mngr_3, info_mngr_2]
                )
            ]
        )

    def test_logical_operator(self):
        """Test logical operator methods"""

        info_mngr_eq = deepcopy(self.info_mngr)
        info_mngr_gt = deepcopy(self.info_mngr)
        info_mngr_gt.evt_dt += timedelta(1)

        assert info_mngr_eq == self.info_mngr
        assert info_mngr_gt > self.info_mngr
        assert self.info_mngr < info_mngr_gt
        assert self.info_mngr != info_mngr_gt


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
            'dt("2018-01-10 00:00:00+00:00")', *args
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
    assert len(
        SearchInfoExpr.eval(
            'tags(("e:1", "r:1"))', *args
        )
    ) > 0
    assert (
        SearchInfoExpr.eval(
            'tags(("e:1", "r:1"))', *args
        ) ==
        SearchInfoExpr.eval(
            'or_(tags("e:1"), tags("r:1"))',
            *args
        )
    )

    # Test serial number
    assert len(
        SearchInfoExpr.eval(
            'srn(("AR-000", "BR-000"))', *args
        )
    ) > 0
    assert (
        SearchInfoExpr.eval(
            'srn(("AR-000", "BR-000"))', *args
        ) ==
        SearchInfoExpr.eval(
            'or_(srn("AR-000"), srn("BR-000"))',
            *args
        )
    )

    # Test and_
    assert (
        SearchInfoExpr.eval(
            'and_(tags("e1"), not_(tags("e1")))', *args
        ) ==
        SearchInfoExpr.eval(
            'not_(all())',
            *args
        )
    )
