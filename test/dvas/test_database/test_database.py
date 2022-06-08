"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.database.database module.

"""

# Import external packages and modules
from copy import deepcopy
from datetime import datetime, timedelta
import pytz
import numpy as np
import pytest

# Import from python packages and modules under test
from dvas.database.model import Prm as TableParameter
from dvas.database.database import InfoManager, InfoManagerMetaData
from dvas.database.database import DBInsertError
from dvas.environ import glob_var
from dvas.hardcoded import TAG_RAW, TAG_GDP


# Define db_data
db_data = {
    'sub_dir': 'test_database',
    'data': [
        {
            'index': np.array([0, 1, 2]),
            'value': np.array([500, 501, 502]),
            'prm_name': 'temp',
            'info': {
                'edt': '20200101T0000Z',
                'mdl_name': 'YT',
                'srn': 'YT-100', 'pid': '0',
                'tags': ('data_test_db', 'e:1', 'r:1', arg_tag),
                'metadata': {'test_key_str': 'one', 'test_key_num': '1'},
                'src': ''
            },
        } for arg_tag in [TAG_RAW, TAG_GDP]
    ]
}


class TestDatabaseManager:
    """Test DatabaseManager class"""

    def test_get_or_none(self, db_init):
        """Test get_or_none method"""

        # Define
        db_mngr = db_init.db_mngr

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
            TableParameter,
            search={
                'where': TableParameter.prm_name.in_(
                    ['dummytst_param1', 'dummytst_param2']
                )
            },
            attr=[[TableParameter.prm_name.name]],
            get_first=False
        ) == [['dummytst_param1'], ['dummytst_param2']]

    def test_add_data(self, db_init):
        """Test add_data method"""

        # Define
        db_mngr = db_init.db_mngr
        data = db_init.data

        # Test add data
        db_mngr.add_data(**data[0], force_write=False)

        # Test add same data (overwrite)
        db_mngr.add_data(**data[0], force_write=True)

        with pytest.raises(DBInsertError):
            db_mngr.add_data(
                [],
                data[0]['value'],
                data[0]['info'],
                data[0]['prm_name'],
            )

        with pytest.raises(DBInsertError):
            db_mngr.add_data(
                data[0]['index'],
                [],
                data[0]['info'],
                data[0]['prm_name'],
            )

        with pytest.raises(DBInsertError):
            db_mngr.add_data(
                data[0]['index'],
                data[0]['value'],
                data[0]['info'],
                'xxxxxxx',
            )

    def test_get_data(self, db_init):
        """Test get_data method"""

        # Define
        db_mngr = db_init.db_mngr
        data = db_init.data

        res = db_mngr.get_data(
            f"and_(dt('{data[0]['info']['edt']}'), srn('{data[0]['info']['srn']}'), tags({data[0]['info']['tags']}))",
            data[0]['prm_name'], True
        )

        assert len(res) > 0
        assert isinstance(res, list)
        assert all([isinstance(arg, dict) for arg in res])
        assert all([arg.keys() == set(['info', 'index', 'value']) for arg in res])
        assert all([isinstance(arg['info'], InfoManager) for arg in res])
        assert all([len(arg['index']) == len(arg['value']) for arg in res])

    def test_get_flgs(self, db_init):
        """Test get_flgs"""

        # Define
        db_mngr = db_init.db_mngr

        assert isinstance(db_mngr.get_flgs(), list)


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
    glob_var.eid_pat = r'e\:\d'
    evt_tag = 'e:1'
    glob_var.rid_pat = r'r\:\d'
    rig_tag = 'r:1'
    metadata = {'key_str': 'one', 'key_num': 1.}
    src = 'test_src'
    info_mngr = InfoManager(
        dt_test, oid_test, tags=[evt_tag, rig_tag],
        metadata=metadata, src=src
    )

    def test_oid(self):
        """Test getting 'oid' attribute"""
        self.info_mngr.oid = self.oid_test
        assert self.info_mngr.oid == sorted(self.oid_test)

    def test_edt(self):
        """Test setting/getting 'edt' attribute"""
        self.info_mngr.edt = self.dt_test
        assert self.info_mngr.edt == datetime(2020, 1, 1, tzinfo=pytz.UTC)

    def test_eid(self):
        """Test getting 'eid' attribute"""
        assert self.info_mngr.eid == self.evt_tag

    def test_rid(self):
        """Test getting 'rid' attribute"""
        assert self.info_mngr.rid == self.rig_tag

    def test_mid(self):
        """Test getting 'mid' attribute"""
        assert self.info_mngr.mid == [arg['mid'] for arg in self.info_mngr.object]

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
        info_mngr_2.edt += timedelta(1)
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
        info_mngr_gt.edt += timedelta(1)

        assert info_mngr_eq == self.info_mngr
        assert info_mngr_gt > self.info_mngr
        assert self.info_mngr < info_mngr_gt
        assert self.info_mngr != info_mngr_gt
