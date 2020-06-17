"""
Examples

"""

# Import
from dvas.data.data import load, update_db
from dvas.dvas_logger import LogManager
from dvas.database.database import db_mngr
from dvas.dvas_helper import TimeIt
from dvas.dvas_environ import glob_var


if __name__ == '__main__':

    # Create database
    db_mngr.create_db()

    with LogManager():
        pass
        update_db('fklpros1')
        update_db('trepros1')
        update_db('prepros1')
        update_db('altpros1')

    data_t1 = load("#e < %2020-01-02T120000Z%", 'trepros1')
    data_t2 = load("#tag_abbr == 'e2'", 'trepros1')
    data_a = load("#tag_abbr == 'b1'", 'altpros1')

    with TimeIt():
        data_t2.resample(inplace=True)
        data_t2.interpolate(inplace=True)
        data_sync = data_t2.synchronise()

    data_t1.plot()

    print(data_t2[0].get_flagged('sync').head())
    print(data_sync[0].get_flagged('sync').head())
