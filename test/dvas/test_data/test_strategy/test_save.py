"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.data.strategy.save module.

"""

# Import from python packages and modules
import pandas as pd
import numpy as np

# Import from python packages and modules under test
from dvas.data.data import MultiProfile
from dvas.database.database import InfoManager
from dvas.data.strategy.data import Profile

# Import from current package
from ...db_fixture import db_init  # noqa pylint: disable=W0611


# Define db_data
db_data = {
    'sub_dir': 'test_strategy_save',
    'data': [
        {
            'type_name': 'YT',
            'srn': 'YT-100',
            'pid': '0',
        },
    ]
}


class TestSave:
    """Test class for Save strategy classes"""

    def test_full_save(self, db_init):
        """ Test the full save strategy chain"""

        data = db_init.data[0]

        # Build some test data by hand
        prf_data = pd.DataFrame({'alt': [10., 15., 20.], 'val': [1., 2., 3.], 'flg': [0, 0, 0]})
        prf_data.set_index([prf_data.index, 'alt'], inplace=True)
        info_mngr = InfoManager('20200303T0303Z', data['oid'], tags='raw')
        prf = Profile(info_mngr, data=prf_data)

        prfs = [prf]

        df_to_db_keys = {'alt': 'altpros1', 'val': 'trepros1', 'flg': None}

        # Create a MultiProfile from scratch, then save it to the db.
        prf_v0 = MultiProfile()
        prf_v0.update(df_to_db_keys, prfs)
        prf_v0.save_to_db(add_tags=['vof1'], rm_tags=['raw'], prms=['val', 'alt'])

        # Now try to load it, and save it back (with different tags)
        prf_v1 = MultiProfile()
        prf_v1.load_from_db("tags('vof1')", 'trepros1', alt_abbr='altpros1', inplace=True)
        prf_v1.save_to_db(add_tags=['vof2'], rm_tags=['vof1'], prms=['val', 'alt'])

        # Now try to fetch that second profile. Do I find it ?
        prf_v2 = MultiProfile()
        prf_v2.load_from_db("tags('vof2')", 'trepros1', alt_abbr='altpros1', inplace=True)

        # Run some simple checks ... but if I got here, I am pretty much ok.
        assert len(prf_v2) == len(prf_v0)
        assert prf_v2.info[0].evt_dt == prf_v0.info[0].evt_dt
        assert np.all(prf_v2.info[0].oid == prf_v0.info[0].oid)
        assert np.all(prf_v2.info[0].oid == prf_v0.info[0].oid)
        assert 'derived' in prf_v2.info[0].tags
        assert 'vof1' not in prf_v2.info[0].tags
