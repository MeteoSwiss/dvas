"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

from pathlib import Path
from dvas.dvas_environ import path_var
from dvas.database.database import DatabaseManager
from dvas.data.data import update_db


test_expl_path = Path(__file__).resolve(strict=True).parents[1] / 'test_examples'

path_var.config_dir_path = test_expl_path / 'config'
path_var.local_db_path = test_expl_path / 'dvas_db'
path_var.orig_data_path = test_expl_path / 'data'


db_mngr = DatabaseManager(create_db=True)
update_db('trepros1')
