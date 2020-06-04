"""
Examples

"""

from dvas.database.database import db_mngr
from dvas.data.data import load, update_db
from dvas.dvas_logger import LogManager
from dvas.dvas_helper import TimeIt

from dvas.config.config import Tag, Flag
from dvas.database.database import OneDimArrayConfigLinker

if __name__ == '__main__':

    inst = OneDimArrayConfigLinker([Tag, Flag])
    a=inst.get_document('Tag')
    print(a)

#    db_mngr.create_db()
#    print(db_mngr)
#    with LogManager():
#        pass
    #     #update_db('fklpros1')
    # update_db('trepros1')
    #     #update_db('prepros1')
    #     #update_db('altpros1')
    #
    #data_t1 = load("tag_abbr == 'e1'", 'trepros1')
    # data_t2 = load("event_id == 'e2'", 'trepros1')
    # #data_a = load("event_id == 'e2'", 'altpros1')
    # #data_p = load("event_id == 'e2'", 'prepros1')
    #
    # with TimeIt():
    #     data_t2.resample(inplace=True)
    #     data_t2.interpolate(inplace=True)
    #     data_sync = data_t2.synchronise()
