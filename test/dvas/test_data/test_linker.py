"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.linker module.

"""

# Import external packages and modules
import numpy as np
from pytest_data import use_data

# Import from python packages and modules
from dvas.data.linker import LocalDBLinker
from dvas.data.linker import CSVHandler, GDPHandler
from dvas.environ import path_var


# Define db_data
db_data = {
    'sub_dir': 'test_linker',
    'data': [
        {
            'index': np.array([0, 1, 2]),
            'value': val,
            'prm_name': 'trepros1',
            'info': {
                'evt_dt': dt,
                'type_name': 'YT',
                'srn': 'YT-100', 'pid': '0',
                'tags': 'load_profile_from_linker',
                'metadata': {},
                'src': ''
            },
        } for val, dt in [
            (np.array([400, 401, 402]), '20200101T0000Z'),
            (np.array([441, 441, 442]), '20200202T0000Z')
        ]
    ]
}


class TestFileHandle:
    """Test FileHandle class"""

    @use_data(db_data={'sub_dir': 'test_filehandle'})
    def test_handle(self):
        """Test handle method"""

        # Define
        csv_handler = CSVHandler()
        gdp_handler = GDPHandler()

        csv_file_path = list(path_var.orig_data_path.rglob('*.csv'))[0]
        gdp_file_path = list(path_var.orig_data_path.rglob('*.nc'))[0]

        res_fmt = {
            'info': dict,
            'prm_name': str,
            'index': np.ndarray, 'value': np.ndarray,
        }

        # Read csv file
        csv_res = csv_handler.handle(csv_file_path, 'trepros1')

        # Test key
        assert isinstance(csv_res, dict)
        assert res_fmt.keys() == csv_res.keys()
        assert all(
            [isinstance(csv_res[key], val)for key, val in res_fmt.items()]
        )

        # Read gdp file
        gdp_res = gdp_handler.handle(gdp_file_path, 'trepros1')

        # Test key
        assert isinstance(gdp_res, dict)
        assert res_fmt.keys() == gdp_res.keys()
        assert all(
            [isinstance(gdp_res[key], val)for key, val in res_fmt.items()]
        )

    @use_data(db_data={'sub_dir': 'test_filehandle'})
    def test_set_next(self):
        """Test set_next method"""

        # Define
        handler1 = CSVHandler()
        res = handler1.set_next(GDPHandler())

        # Test return
        assert isinstance(res, GDPHandler)


class TestLoadDBLinker:
    """Test LoadDBLinker class"""

    def test_load(self, db_init):
        """Test load method"""

        # Define
        db_linker = LocalDBLinker()
        data = db_init.data

        # Init
        res = db_linker.load("all()", data[0]['prm_name'])

        # Test
        assert isinstance(res, list)

    def test_save(self, db_init):
        """Test save method"""

        # Define
        db_linker = LocalDBLinker()
        data = db_init.data

        # Save data
        db_linker.save(
            [{
                'index': data[0]['index'],
                'value': data[0]['value'],
                'info': data[0]['info'],
                'prm_name': data[0]['prm_name'],
            }]
        )

        # Force to save same data
        db_linker.save(
            [{
                'index': data[0]['index'],
                'value': data[0]['value'],
                'info': data[0]['info'],
                'prm_name': data[0]['prm_name'],
                'force_write': True
            }]
        )
