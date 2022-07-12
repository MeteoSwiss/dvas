"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.linker module.

"""

# Import external packages and modules
import inspect
import numpy as np
import pandas as pd
from pytest_data import use_data

# Import from python packages and modules
from dvas.data.linker import LocalDBLinker
from dvas.data.linker import CSVHandler, GDPHandler
from dvas.data.linker import GetreldtExpr
from dvas.environ import path_var
from dvas.config.config import OrigData

# Define db_data
db_data = {
    'sub_dir': 'test_linker',
    'data': [
        {
            'index': np.array([0, 1, 2]),
            'value': val,
            'prm_name': 'temp',
            'info': {
                'edt': dt,
                'mdl_name': 'YT',
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


def test_pandas_csv_read_args():
    """ Test dedicated to error 160, when pandas inspect package could no longer extract the
    arguments of pandas.read_csv(). This is used to define the content of PD_CSV_READ_ARGS in
    linker.py """

    assert len(list(inspect.signature(pd.read_csv).parameters.keys())[1:]) > 0


class TestFileHandle:
    """Test FileHandle class"""

    # Init orig data config
    origdata_config_mngr = OrigData()
    origdata_config_mngr.read()

    @use_data(db_data={'sub_dir': 'test_filehandle'})
    def test_handle(self):
        """Test handle method"""

        # Define
        csv_handler = CSVHandler(self.origdata_config_mngr)
        gdp_handler = GDPHandler(self.origdata_config_mngr)

        csv_file_path = list(path_var.orig_data_path.rglob('*.csv'))[0]
        gdp_file_path = list(path_var.orig_data_path.rglob('*.nc'))[0]

        res_fmt = {
            'info': dict,
            'prm_name': str,
            'index': np.ndarray, 'value': np.ndarray,
        }

        # Read csv file
        csv_res = csv_handler.handle(csv_file_path, 'temp')

        # Test key
        assert isinstance(csv_res, dict)
        assert res_fmt.keys() == csv_res.keys()
        assert all(
            [isinstance(csv_res[key], val)for key, val in res_fmt.items()]
        )

        # Read gdp file
        gdp_res = gdp_handler.handle(gdp_file_path, 'temp')

        # Test key
        assert isinstance(gdp_res, dict)
        assert res_fmt.keys() == gdp_res.keys()
        assert all(
            [isinstance(gdp_res[key], val) for key, val in res_fmt.items()]
        )

        # Test load of missing parameter
        res_missing = gdp_handler.handle(gdp_file_path, 'temp_missing')
        assert len(res_missing['value']) == 0

    @use_data(db_data={'sub_dir': 'test_filehandle'})
    def test_set_next(self):
        """Test set_next method"""

        # Define
        handler1 = CSVHandler(self.origdata_config_mngr)
        res = handler1.set_next(GDPHandler(self.origdata_config_mngr))

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


class TestGetreldtExpr:
    """ Test the GetreldtExpr class """

    @staticmethod
    def basic_setup(round_lvl):
        """ Setup a basic routine that can be repeated to test different rounding levels. """

        # Feed the class init
        item = GetreldtExpr(
            ['2022-07-12T11:14:00.050Z', '2022-07-12T11:14:01.061Z', '2022-07-12T11:14:02.050Z'],
            fmt='%Y-%m-%dT%H:%M:%S.%fZ', round_lvl=round_lvl)

        # Abuse the class parent elements to read in the data directly
        item._FCT = pd.Series
        item._ARGS = {}
        item._KWARGS = {}

        # Process the Series of datetime strings
        return item.interpret()

    def test_rounding(self):
        """ Test that rounding of microseconds works as intended. """

        assert all((self.basic_setup(None)).round(decimals=3) == np.array([0, 1.011, 2]))
        assert all((self.basic_setup(1)).round(decimals=3) == np.array([0, 1.000, 2]))
        assert all((self.basic_setup(2)).round(decimals=3) == np.array([0, 1.010, 2]))
        assert all((self.basic_setup(3)).round(decimals=3) == np.array([0, 1.011, 2]))
