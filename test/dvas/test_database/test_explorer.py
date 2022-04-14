"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.database.explorer module.

"""

# Import external packages and modules
import numpy as np
import pytest

# Import from python packages and modules under test
from dvas.database.explorer import ReadDatabase
from dvas.hardcoded import TAG_RAW_NAME, TAG_GDP_NAME
from dvas.errors import SearchError

# Define db_data
db_data = {
    'sub_dir': 'test_explorer',
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
        } for arg_tag in [TAG_RAW_NAME, TAG_GDP_NAME]
    ]
}


class TestReadDatabase:
    """Test ReadDatabase"""

    def test_info(self):
        """Test info method"""

        # Init
        reader = ReadDatabase()

        # Test
        assert isinstance(reader.info('all()'), list)
        assert isinstance(reader.info('all() -r'), list)
        assert isinstance(reader.info('all() -l'), int)

        # Test help
        assert reader.info('?') is None
        assert reader.info('-h') is None

        with pytest.raises(SearchError):
            reader.info('dummy')

    def test_prm(self):
        """Test prm method"""

        # Init
        reader = ReadDatabase()

        # Test
        assert isinstance(reader.prm('all()'), list)
        assert isinstance(reader.prm('all() -r'), list)
        assert isinstance(reader.prm('all() -l'), int)

        with pytest.raises(SearchError):
            reader.prm('dummy')

    def test_obj(self):
        """Test obj method"""

        # Init
        reader = ReadDatabase()

        # Test
        assert isinstance(reader.obj('all()'), list)
        assert isinstance(reader.obj('all() -r'), list)
        assert isinstance(reader.obj('all() -l'), int)

        with pytest.raises(SearchError):
            reader.obj('dummy')
