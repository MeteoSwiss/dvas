"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

# Import external packages and modules
from pathlib import Path

# Import from current python packages and modules
from dvas.environ import path_var

# Set expl_path
test_expl_path = Path(__file__).resolve(strict=True).parent / 'processing_arena'

# Set config and data expl path
# TODO
#  Use a fixture for config and data
path_var.config_dir_path = test_expl_path / 'config'
path_var.orig_data_path = test_expl_path / 'data'
path_var.output_path = test_expl_path / 'output'
