"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Loader strategies

"""

# Import from external packages
from abc import ABCMeta, abstractmethod

# Import from current package
from .data import ProfileManager
from ..linker import LocalDBLinker


class LoadDataStrategy(metaclass=ABCMeta):
    """Abstract class to manage data loading strategy"""

    def __init__(self):
        self._db_linker = LocalDBLinker()

    @abstractmethod
    def load(self, *args, **kwargs):
        """Strategy required method"""


class LoadTimeDataStrategy(LoadDataStrategy):
    """Class to manage loading of same time data parameter"""

    def load(self, search, prm_abbr):
        """Implementation of load method"""

        # Get data
        res = self._db_linker.load(search, prm_abbr)

        # Test
        assert len(res) > 0, "No data"

        # Load data
        out = [ProfileManager(data['event'], data['data']) for data in res]

        return {'data': out}


class LoadAltDataStrategy(LoadTimeDataStrategy):
    """Class to manage loading of same time data parameter"""

    def load(self, search, prm_abbr, alt_prm_abbr):
        """Implementation of load method"""

        # Load prm_abbr
        out_prm = super().load(search, prm_abbr)

        # Load alt
        dt_str = f"%{out_prm['data'][0].event_mngr.event_dt:%Y%m%dT%H:%M:%SZ}%"
        sn_str = f"'{out_prm['data'][0].event_mngr.sn}'"
        search_alt = "{_dt == %s} & {_sn == %s}" % (dt_str, sn_str)
        alt_res = self._db_linker.load(search_alt, alt_prm_abbr)

        # Test
        assert len(alt_res) == 1, "No or too many alt data"

        out_alt_prm = ProfileManager(alt_res[0]['data'], alt_res[0]['event'])

        return dict({'alt': out_alt_prm}, **out_prm)
