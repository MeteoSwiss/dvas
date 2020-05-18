"""
Examples

"""

from dvas.database.database import db_mngr
from dvas.data.data import load, update_db


if __name__ == '__main__':

    db_mngr.create_db()

    update_db('trepros1')
    update_db('prepros1')
    update_db('altpros1')
    data_t1 = load("event_id == 'e1'", 'trepros1')
    data_t2 = load("event_id == 'e2'", 'trepros1')
    data_a = load("event_id == 'e2'", 'altpros1')
    data_p = load("event_id == 'e2'", 'prepros1')
    data_t2.resample(inplace=True)
    data_t2.interpolate(inplace=True)
    data_sync = data_t2.synchronise()
