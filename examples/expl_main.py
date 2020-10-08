"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: examples
"""



# Import
import numpy as np
from dvas.data.data import TemporalMultiProfileManager
from dvas.data.data import AltitudeMultiProfileManager
from dvas.data.data import update_db
from dvas.dvas_logger import LogManager
from dvas.database.database import db_mngr
from dvas.database import database as db

from dvas.data.math import calc_tropopause, calc_tropopause_old
from dvas.dvas_helper import TimeIt


if __name__ == '__main__':

    # with db.DBAccess(db.db) as _:
    #     a = db.AndExpr(db.NotExpr(db.TagExpr('empty')), db.TagExpr('e1'))
    #     a = db.AndExpr(a, db.ParameterExpr('trepros1'))
    #     print(a.interpret())
    #
    #     # a = db.AbstractExpression.eval(
    #     #     "tag('e1')",
    #     # )
    #     # print(a)

    """

    # Create database
    db_mngr.create_db()

    # Update DB + log
    with LogManager():
        update_db('trepros1', strict=True)
        update_db('treprosu_')
        update_db('altpros1')
        update_db('prepros1')

    """

    filter = "tag('e1')"


    # Time
    data_t = TemporalMultiProfileManager.load(filter, 'trepros1')

    """


    data_s = data_t.sort()
    data_r = data_s.resample()
    data_sy = data_r.synchronize()
    # #data_sy.plot()
    data_sy.save({'data': 'dummy_3'})
    test = TemporalMultiProfileManager.load(filter, 'dummy_3')
    test = test.sort()
    print(
        [[np.max(np.abs(arg[0] - arg[1]))
         for arg in zip(data_sy.values[key], test.values[key])
         ] for key in data_sy.keys]
    )
    data_sy.plot()

    
    # Alt
    data_t = AltitudeMultiProfileManager.load(
        filter, 'trepros1', 'altpros1'
    )
    data_s = data_t.sort()
    data_r = data_s.resample()
    data_sy_t = data_r.synchronize(method='time')
    data_sy_a = data_r.synchronize(method='alt')
    data_sy_t.save({'data': 'dummy_0', 'alt': 'dummy_1'})

    test = data_t = AltitudeMultiProfileManager.load(
        filter, 'dummy_0', 'dummy_1'
    )
    test = test.sort()
    print(
        [[np.max(np.abs(arg[0] - arg[1]))
         for arg in zip(data_sy_t.values[key], test.values[key])
         ] for key in data_sy_t.keys]
    )
    data_sy_t.plot()
    """


