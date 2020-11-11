"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.save module.

"""

# Import from python packages and modules
import pytest
import datetime
import pytz
from collections import OrderedDict
import pandas as pd

from dvas.data.data import MultiProfile
from dvas.database.database import EventManager
from dvas.data.strategy.save import SaveDataStrategy
from dvas.data.strategy.data import Profile
from dvas.data.data import MultiProfile

class TestSave:
    """Test class for Save strategy classes"""

    # Build some test data by hand
    data = pd.DataFrame({'alt': [10., 15., 20.], 'val': [1., 2., 3.], 'flg':[0, 0, 0]})
    evt = EventManager(datetime.datetime.now(tz=pytz.UTC), '123456789', tag_abbr={'raw'})
    prf = Profile(evt, data=data)

    values = {'prf':[data]}
    events = {'prf':[evt]}

    df_to_db_keys = {'prf':{'alt': 'altpros1', 'val': 'trepros1'}}

    save_stgy = SaveDataStrategy()


    # TODO: this will fail until flags are allowed in the DB.
    #def test_simple_save(self):
    #    """Test simple save startegy"""
    #
    #    # Try to save the mock data
    #    self.save_stgy.save(self.values, self.events, self.df_to_db_keys,
    #                        add_tags=['vof'], rm_tags=None)
    #
    #    # If I get here without trouble, then succeeed ?
    #    assert True
    #

    def test_full_save(self):
        """ Test the full save strategy chain"""

        filt_gdp = "tag('gdp')"
        filt_raw = "tag('raw')"
        filt_vof = "tag('vof')"
        filt_vof2 = "tag('vof2')"

        filt_in = "and_(%s,%s)" % (filt_gdp, filt_raw)

        prf_v0 = MultiProfile()
        prf_v0.DB_VARIABLES = {'prf':{'alt':'altpros1', 'val':'trepros1'}}
        prf_v0.profiles['prf']=[self.prf]
        prf_v0.events['prf']=[self.evt]
        #prf.load(filt_raw, 'trepros1', alt_abbr='altpros1', inplace=True)
        prf_v0.save(add_tags=['vof'], rm_tags=['raw'], prm_list=['val', 'alt'])

        prf_v1 = MultiProfile()
        prf_v1.load(filt_vof, 'trepros1', alt_abbr='altpros1', inplace=True)
        prf_v1.save(add_tags=['vof2'], rm_tags=None, prm_list=['val', 'alt'])

        # TODO: this is a bug I do not understand:
        # If I save the same profile 2 times in a row with different tags, it works the first time,
        # but faisl the second time. Why is that ?
        prf_v2 = MultiProfile()
        prf_v2.load(filt_vof2, 'trepros1', alt_abbr='altpros1', inplace=True)

        assert False
