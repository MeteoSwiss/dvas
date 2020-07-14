"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: examples
"""

# Import
from dvas.data.data import load, update_db
from dvas.dvas_logger import LogManager
from dvas.database.database import db_mngr
from dvas.dvas_helper import TimeIt


if __name__ == '__main__':

    # Create database
    db_mngr.create_db()

    # Update DB + log
    with LogManager():
        update_db('trepros1')
        update_db('treprosu')

    # Update all parameters ending with 1 + log
    #with LogManager():
    #    update_db('%1')

    # Same without log
    #update_db('%1')

    # Load
    #data_t1 = load("#e < %2020-01-02T120000Z%", 'trepros1')
    data_t2 = load("#tag == 'e1'", 'trepros1')
    data_a = load("#tag_abbr == 'b1'", 'altpros1')
    data_gdp = load("#tag == 'gdp'", 'trepros1')

    with TimeIt():
        data_t2.resample(inplace=True)
        data_t2.interpolate(inplace=True)
        data_sync = data_t2.synchronise()

    data_t2.plot()
    data_sync.plot()
