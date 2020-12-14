"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""


from pathlib import Path

from dvas.data.data import update_db
from dvas.environ import path_var
from dvas.database.database import DatabaseManager

test_expl_path = Path(__file__).resolve(strict=True).parent / 'processing_arena'

path_var.config_dir_path = test_expl_path / 'config'
# TODO
#  Use a session tmpdir to create the database
path_var.local_db_path = test_expl_path / 'dvas_db'
path_var.orig_data_path = test_expl_path / 'data'

# Create database
# This is by no means a satisfactory solution.
# It must be possible to create an independent DB for each test
# and not a global DB.
DatabaseManager(reset_db=True)
update_db('trepros1', strict=True)
