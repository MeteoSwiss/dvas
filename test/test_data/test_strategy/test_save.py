"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.save module.

"""

# Import from python packages and modules
import datetime
import pytz
import pandas as pd
import numpy as np

from dvas.data.linker import LocalDBLinker
from dvas.data.data import MultiProfile
from dvas.database.database import InfoManager
from dvas.data.strategy.save import SaveDataStrategy
from dvas.data.strategy.data import Profile
from dvas.data.data import MultiProfile

class TestSave:
    """Test class for Save strategy classes"""

    # Build some test data by hand
    data = pd.DataFrame({'alt': [10., 15., 20.], 'val': [1., 2., 3.], 'flg':[0, 0, 0]})
    data.set_index([data.index, 'alt'])
    evt = InfoManager(datetime.datetime.now(tz=pytz.UTC), '123456789', tags={'raw'})
    prf = Profile(evt, data=data)

    prfs = [prf]
    events = [evt]

    df_to_db_keys = {'alt': 'altpros1', 'val': 'trepros1', 'flg': None}

    def test_full_save(self):
        """ Test the full save strategy chain"""

        # Create a MultiProfile from scratch, then save it to the db.
        prf_v0 = MultiProfile()
        prf_v0.update(self.df_to_db_keys, self.prfs)
        #prf.load(filt_raw, 'trepros1', alt_abbr='altpros1', inplace=True)
        prf_v0.save_to_db(add_tags=['vof1'], rm_tags=['raw'], prms=['val', 'alt'])

        # Now try to load it, and save it back (with different tags)
        prf_v1 = MultiProfile()
        prf_v1.load_from_db("tag('vof1')", 'trepros1', alt_abbr='altpros1', inplace=True)
        prf_v1.save_to_db(add_tags=['vof2'], rm_tags=['vof1'], prms=['val', 'alt'])

        # Now try to fetch that second profile. Do I find it ?
        prf_v2 = MultiProfile()
        prf_v2.load_from_db("tag('vof2')", 'trepros1', alt_abbr='altpros1', inplace=True)

        # Run some simple checks ... but if I got here, I am pretty much ok.
        assert len(prf_v2) == len(prf_v0)
        assert prf_v2.info[0].evt_dt == prf_v0.info[0].evt_dt
        assert np.all(prf_v2.info[0].srn == prf_v0.info[0].srn)
        assert np.all(prf_v2.info[0].srn == prf_v0.info[0].srn)
        assert 'derived' in prf_v2.info[0].tags
        assert 'vof1' not in prf_v2.info[0].tags
