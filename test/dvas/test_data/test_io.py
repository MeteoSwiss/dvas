"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.io module.

"""

# Import from python packages and modules
from dvas.data.io import update_db
from dvas.data.strategy.load import LoadRSProfileStrategy
from dvas.hardcoded import TAG_ORIGINAL


# Define db_data
db_data = {
    'sub_dir': 'test_strategy_load',
}


def test_update_db():
    """Test update_db function"""

    # Update
    update_db('temp', strict=True)
    update_db('temp_flag', strict=True)
    update_db('time', strict=True)
    update_db('gph', strict=True)

    # Load
    prf_stgy = LoadRSProfileStrategy()
    data = prf_stgy.execute(
        f"tags('{TAG_ORIGINAL}')", 'temp', 'time',
        alt_abbr='gph'
    )

    assert all([not arg.val.isna().all() for arg in data[0]])
    assert all([not arg.alt.isna().all() for arg in data[0]])
    assert all([not arg.tdt.isna().all() for arg in data[0]])
    assert sum([not arg.flg.isna().all() for arg in data[0]]) == 2
