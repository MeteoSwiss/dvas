"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.data module.

"""

# Import from python packages and modules
from pathlib import Path
import pytest

from dvas.data.strategy.data import FlagManager

# Define
current_pkge_path = Path(__file__).parent
ok_fixture_dir = current_pkge_path / 'etc' / 'ok'
ko_fixture_dir = current_pkge_path / 'etc' / 'ko'


class TestFlagManager:
    """Test class for FlagManager"""

    flag_mngr = FlagManager()



@pytest.mark.datafiles(ok_fixture_dir.as_posix())
def test_instantiate_config_managers(datafiles):
