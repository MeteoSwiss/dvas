"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Loader strategies

"""

# Import from external packages
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np

# Import from current package
from .data import Profile, RSProfile, GDPProfile
from ..linker import LocalDBLinker
from ...database.model import Data
from ...dvas_logger import dvasError


# Define
INDEX_NM = Data.index.name
VALUE_NM = Data.value.name


class LoadProfileStrategy(metaclass=ABCMeta):
    """Base class to manage the data loading strategy of Profile instances."""

    def __init__(self):
        self._db_linker = LocalDBLinker()

    def __fetch__(self, search, abbrs):
        """ A base function that fetches data from the database.

        Args:
            search (str): selection criteria
            abbrs (dict): dict of id: 'parameter names' to extract.

        Returns:
            list of pd.DataFrame: a DataFrame for each of the events found.

        Exemple:
            ```
            import dvas.data.strategy.load as ld
            t = ld.LoadProfileStrategy()
            t.__fetch__("dt('20160715T120000Z', '==')", {'alt':'altpros1', 'val':'trepros1'})
            ```

        """

        n_evts = 0

        # Loop through the requested parameters and extract them from the database.
        for i, item in enumerate(abbrs):

            # First of all, ignore
            if abbrs[item] is None:
                continue

            # Query the db
            res = self._db_linker.load(search, abbrs[item])

            # Make sure some data was returned.
            assert len(res) > 0, \
                "No %s data found with in the database with criteria: %s" % (abbrs[item], search)

            if i == 0:
                # Since the order through which abbreviations are queried will vary, keep track
                # of the first event manually.
                ref_evts = [item['event'] for item in res]
                # How many events do I get ?
                n_evts = len(ref_evts)
                #If I know how many events I have, I can create a suitable number of DataFrame to
                # store the related data
                out = [pd.DataFrame() for _ in range(n_evts)]

            # Now loop through each event returned
            for j, jtem in enumerate(res):

                # Check that the order of events that was returned indeed matches that of the
                # first parameter queried.
                if jtem['event'] != ref_evts[j]:
                    raise dvasError('Bad fetch order from the database.')

                # Very well, let us add this data to the appropriate DataFrame
                out[j][item] = jtem['data']['value']

        return (ref_evts, out)


    def load(self, search, prm_abbr, alt_abbr=None):
        """ Load method to fetch data from the databse.

        Args:
            search (str): selection criteria
            prm_abbr (str): name of the parameter values to extract
            alt_abbr (str, optional): name of the altitude parameter to extract. Dafaults to None.
        """

        # Fetch the data from the database
        (evts, out) = self.__fetch__(search, {'val':prm_abbr, 'alt':alt_abbr})

        # Deal with the None
        if alt_abbr is None:
            out = [pd.concat([item, pd.Series(np.nan, name='alt')], axis=1) for item in out]

        #TODO: load flags from the DB
        out = [pd.concat([item, pd.Series([0]*len(item), dtype=np.int64, name='flg')], axis=1)
               for item in out]

        out = [Profile(evts[j], data=out[j]) for j in range(len(evts))]

        return {'prf': out}

class LoadRSProfileStrategy(LoadProfileStrategy):
    """Child class to manage the data loading strategy of RSProfile instances."""

    def load(self, search, prm_abbr, alt_abbr=None, tdt_abbr=None):
        """ Load method to fetch data from the databse.

        Args:
            search (str): selection criteria
            prm_abbr (str): name of the parameter values to extract
            alt_abbr (str, optional): name of the altitude parameter to extract. Dafaults to None.
            tdt_abbr (str, optional): name of the time delta parameter to extract. Defaults to None.

        """

        # Fetch the data from the database
        (evts, out) = self.__fetch__(search, {'val':prm_abbr, 'alt':alt_abbr, 'tdt':tdt_abbr})

        # Deal with the None
        if alt_abbr is None:
            out = [pd.concat([item, pd.Series(np.nan, name='alt')], axis=1) for item in out]

        if tdt_abbr is None:
            out = [pd.concat([item, pd.Series(np.nan, dtype='timedelta64[ns]', name='tdt')], axis=1)
                   for item in out]

        #TODO: load flags from the DB
        out = [pd.concat([item, pd.Series([0]*len(item), dtype=np.int64, name='flg')], axis=1)
               for item in out]

        out = [RSProfile(evts[j], data=out[j]) for j in range(len(evts))]

        return {'rs_prf': out}


class LoadGDPProfileStrategy(LoadProfileStrategy):
    """Child class to manage the data loading strategy of GDPProfile instances."""

    def load(self, search, prm_abbr, alt_abbr=None, tdt_abbr=None,
             ucn_abbr=None, ucr_abbr=None, ucs_abbr=None, uct_abbr=None):
        """ Load method to fetch data from the databse.

        Args:
            search (str): selection criteria
            prm_abbr (str): name of the parameter values to extract
            alt_abbr (str, optional): name of the altitude parameter to extract. Dafaults to None.
            tdt_abbr (str, optional): name of the time delta parameter to extract. Defaults to None.
            ucn_abbr (str, optional): name of the true un-correlated uncertainty parameter to
               extract. Defaults to None.
            ucr_abbr (str, optional): name of the true rig un-correlated uncertainty parameter to
               extract. Defaults to None.
            ucs_abbr (str, optional): name of the true spatial-correlated uncertainty parameter to
               extract. Defaults to None.
            uct_abbr (str, optional): name of the true time-correlated uncertainty parameter to
               extract. Defaults to None.

        """

        # Fetch the data from the database
        (evts, out) = self.__fetch__(search, {'val':prm_abbr, 'alt':alt_abbr, 'tdt':tdt_abbr,
                                              'ucn':ucn_abbr, 'ucr':ucr_abbr, 'ucs':ucs_abbr,
                                              'uct':uct_abbr})

        # Deal with the None
        if alt_abbr is None:
            out = [pd.concat([item, pd.Series(np.nan, name='alt')], axis=1) for item in out]
        if tdt_abbr is None:
            out = [pd.concat([item, pd.Series(np.nan, dtype='timedelta64[ns]', name='tdt')], axis=1)
                   for item in out]
        if ucn_abbr is None:
            out = [pd.concat([item, pd.Series(np.nan, name='ucn')], axis=1) for item in out]
        if ucr_abbr is None:
            out = [pd.concat([item, pd.Series(np.nan, name='ucr')], axis=1) for item in out]
        if ucs_abbr is None:
            out = [pd.concat([item, pd.Series(np.nan, name='ucs')], axis=1) for item in out]
        if uct_abbr is None:
            out = [pd.concat([item, pd.Series(np.nan, name='uct')], axis=1) for item in out]

        #TODO: load flags from the DB
        out = [pd.concat([item, pd.Series([0]*len(item), dtype=np.int64, name='flg')], axis=1)
               for item in out]

        out = [GDPProfile(evts[j], data=out[j]) for j in range(len(evts))]

        return {'gdp_prf': out}
