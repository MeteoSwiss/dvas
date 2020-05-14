

from pathlib import Path


from jsonschema import validate
import re

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import IndexDateFormatter

from dvas.database.database import db_mngr, EventManager, Flag, ConfigLinker
from dvas.database.database import Instrument, InstrType
from dvas.database.database import EventManager as e_mngr
from dvas.database.model import Parameter

from dvas.config import config
from dvas.data.linker import OriginalCSVLinker
from dvas.data.data import load, update_db
from dvas.data.data import FlagManager

from dvas.dvas_helper import TimeIt


doc = """
meas_site: PAY_RS
flight: f1
batch: b1
instrument: i1
datetime: 2020-01-02T000000Z
"""



if __name__ == '__main__':

    db_mngr.create_db()


    # import numpy as np
    #
    # n_data = 7000
    #
    # instr_ids = ['i' + str(i) for i in range(1, 3)]
    #
    # data_raw = [pd.Series(np.random.randn(n_data)*10).round(4) for instr_id in instr_ids]
    #
    # event_dt = pd.to_datetime('2020.03.10T00:00:00Z')
    #
    # args_list = [
    #     {'data': data_raw[i],
    #      'event': EventManager(
    #         event_dt=event_dt,
    #         instr_id=instr_id,
    #         prm_abbr='trepros1',
    #         batch_id='b0',
    #         day_event=False
    #         )
    #     } for i, instr_id in enumerate(instr_ids)
    # ]
    #
    #
    # with TimeIt():
    #     for args in args_list:
    #         db_mngr.add_data(**args)
    #
    # with TimeIt():
    #     d = db_mngr.get_data(f"event_dt == '{event_dt}'", 'trepros1')


    # inst = config.OrigData()
    # inst.read()
    # print(inst)
    # print('***')
    # print(inst[['_vai-rs41', '_trepros1', '_i1', 'idx_unit']])

    update_db('trepros1')
    # update_db('prepros1')
    # update_db('altpros1')
    data_t1 = load("event_id == 'e1'", 'trepros1')
    #data_t2 = load("event_id == 'e2'", 'trepros1')
    # data_a = load("event_id == 'e2'", 'altpros1')
    # data_p = load("event_id == 'e2'", 'prepros1')
    # data.resample(inplace=True)
    # data.interpolate(inplace=True)
    # data_sync = data.synchronise()


