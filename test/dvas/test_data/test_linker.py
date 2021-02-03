"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.linker module.

"""

# Import external packages and modules
import numpy as np

# Import from python packages and modules
from dvas.data.linker import LocalDBLinker
from dvas.data.linker import CSVHandler, GDPHandler
from dvas.environ import path_var
from dvas.database.database import InfoManager


class TestFileHandle:
    """Test FileHandle class"""

    csv_handler = CSVHandler()
    gdp_handler = GDPHandler()

    csv_file_path = list(path_var.orig_data_path.rglob('*.csv'))[0]
    gdp_file_path = list(path_var.orig_data_path.rglob('*.nc'))[0]

    res_fmt = {
        'info': dict,
        'prm_name': str,
        'index': np.ndarray, 'value': np.ndarray,
        'source_info': str
    }

    def test_handle(self):
        """Test handle method"""

        # Read csv file
        csv_res = self.csv_handler.handle(self.csv_file_path, 'trepros1')

        # Test key
        assert isinstance(csv_res, dict)
        assert self.res_fmt.keys() == csv_res.keys()
        assert all(
            [isinstance(csv_res[key], val)for key, val in self.res_fmt.items()]
        )

        # Read gdp file
        gdp_res = self.gdp_handler.handle(self.gdp_file_path, 'trepros1')

        # Test key
        assert isinstance(gdp_res, dict)
        assert self.res_fmt.keys() == gdp_res.keys()
        assert all(
            [isinstance(gdp_res[key], val)for key, val in self.res_fmt.items()]
        )

    def test_set_next(self):
        """Test set_next method"""

        # Define
        handler1 = CSVHandler()
        res = handler1.set_next(GDPHandler())

        # Test return
        assert isinstance(res, GDPHandler)


class TestLoadDBLinker:
    """Test LoadDBLinker class"""

    # Define
    db_linker = LocalDBLinker()
    index = np.arange(3)
    values = np.array([440, 441, 442])
    prm = 'trepros1'
    sn = 'YT-100'
    info = InfoManager.from_dict(
        {
            'evt_dt': '20200101T0000Z',
            'srn': sn, 'pid': '0',
            'tags': 'load_profile_from_linker',
            'metadata': {},
        }
    )

    def test_load(self):
        """Test load method"""

        # Init
        res = self.db_linker.load("all()", self.prm)

        # Test
        assert isinstance(res, list)

    def test_save(self):
        """Test save method"""

        # Save data
        self.db_linker.save(
            [{
                'index': self.index,
                'value': self.values,
                'info': self.info,
                'prm_name': self.prm,
                'source_info': 'test_add_data'
            }]
        )

        # Force to save same data
        self.db_linker.save(
            [{
                'index': self.index,
                'value': self.values,
                'info': self.info,
                'prm_name': self.prm,
                'source_info': 'test_add_data',
                'force_write': True
            }]
        )
