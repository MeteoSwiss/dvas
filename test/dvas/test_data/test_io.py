"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.io module.

"""

# Import from python packages and modules
from dvas.data.io import update_db
from dvas.data.strategy.load import LoadRSProfileStrategy


# Define db_data
db_data = {
    'sub_dir': 'test_strategy_load',
}


def test_update_db():
    """Test update_db function"""

    # Update
    update_db('trepros1', strict=True)
    update_db('trepros1_flag', strict=True)
    update_db('tdtpros1', strict=True)
    update_db('altpros1', strict=True)

    # Load
    prf_stgy = LoadRSProfileStrategy()
    data = prf_stgy.execute(
        "tags('raw')", 'trepros1', 'tdtpros1',
        alt_abbr='altpros1'
    )

    assert all([not arg.val.isna().all() for arg in data[0]])
    assert all([not arg.alt.isna().all() for arg in data[0]])
    assert all([not arg.tdt.isna().all() for arg in data[0]])
    assert sum([not arg.flg.isna().all() for arg in data[0]]) == 2
