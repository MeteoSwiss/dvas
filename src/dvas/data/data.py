"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data management

"""

# Import from external packages
from copy import deepcopy
import numpy as np
import pandas as pd

# Import from current package
from .linker import LocalDBLinker, CSVHandler, GDPHandler
from ..plot.plot import basic_plot
from .math import crosscorr
from ..database.database import db_mngr
from ..database.model import Flag, Parameter
from ..database.database import OneDimArrayConfigLinker
from ..dvas_logger import localdb, rawcsv
from ..dvas_environ import path_var
from ..config.definitions.tag import TAG_RAW_VAL, TAG_DERIVED_VAL, TAG_EMPTY_VAL


# Define
FLAG = 'flag'
VALUE = 'value'
cfg_linker = OneDimArrayConfigLinker()


class FlagManager:
    """Flag manager class"""

    FLAG_BIT_NM = Flag.bit_number.name
    FLAG_ABBR_NM = Flag.flag_abbr.name
    FLAG_DESC_NM = Flag.flag_desc.name

    def __init__(self, index):
        """
        Args:
            index (iterable): Flag index
        """
        self._flags = {
            arg[self.FLAG_ABBR_NM]: arg
            for arg in db_mngr.get_flags()
        }

        self._data = pd.Series(
            np.zeros(len(index),), index=index, dtype=int
        )

    def __len__(self):
        return len(self.data)

    @property
    def flags(self):
        """dict: Flag abbr, description and bit position."""
        return self._flags

    @property
    def data(self):
        """pandas.Series: Data corresponding flag value."""
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def reindex(self, index, set_abbr):
        """Reindex method

        Args:
            index (iterable): New index
            set_abbr (str): Flag abbr used to replace NaN creating during
                reindexing.

        """

        # Reindex flag
        self.data = self.data.reindex(index)

        # Set resampled flag
        self.set_bit_val(set_abbr, True, self.data.isna())

    def get_bit_number(self, abbr):
        """Get bit number corresponding to given flag abbr"""
        return self.flags[abbr][self.FLAG_BIT_NM]

    def set_bit_val(self, abbr, set_val, index=None):
        """Set data flag value to one

        Args:
            abbr (str):
            set_val (bool): Default to True
            index (pd.Index, optional): Default to None.

        """

        # Define
        def set_to_true(x):
            """Set bit to True"""
            if np.isnan(x):
                out = (1 << self.get_bit_number(abbr))
            else:
                out = int(x) | (1 << self.get_bit_number(abbr))

            return out

        def set_to_false(x):
            """Set bit to False"""
            if np.isnan(x):
                out = 0
            else:
                out = int(x) & ~(1 << self.get_bit_number(abbr))

            return out

        # Init
        if index is None:
            index = self.data.index

        # Set bit
        if set_val is True:
            self._data.loc[index] = self.data.loc[index].apply(set_to_true)
        else:
            self._data.loc[index] = self.data.loc[index].apply(set_to_false)

    def get_bit_val(self, abbr):
        """Get data flag value

        Args:
            abbr (str):

        """
        bit_nbr = self.get_bit_number(abbr)
        return self.data.apply(lambda x: (x >> bit_nbr) & 1)


class EmptyProfileManager:
    """Empty profile"""

    def __init__(self, event_mngr):
        """

        Args:
            event_mngr:
        """

        # Set attributes
        self._event_mngr = event_mngr

    @property
    def data(self):
        """pd.Series: Data"""
        return pd.Series()

    @property
    def event_mngr(self):
        """EventManager: Corresponding data event manager"""
        return self._event_mngr


class ProfileManger(EmptyProfileManager):
    """Profile manager"""

    def __init__(self, data, event_mngr):
        """Constructor

        Args:
            data (pd.Series): pd.Series with any index
            event_mngr (EventManager):

        """

        super().__init__(event_mngr)

        # Test
        self.check_data(data)

        # Set attributes
        self._data = data
        self._data.name = None
        self._flag_mngr = FlagManager(data.index)

    @property
    def data(self):
        """pd.Series: Data"""
        return self._data

    @data.setter
    def data(self, val):
        # Test
        self.check_data(val)
        assert len(val) == len(self.flag_mngr), 'len(data) != len(flag)'

        # Set data
        self._data = val

        # Modify tag 'raw' -> 'derived'
        self.event_mngr.rm_tag(TAG_RAW_VAL)
        self.event_mngr.add_tag(TAG_DERIVED_VAL)

    @property
    def flag_mngr(self):
        """FlagManager: Flag manager"""
        return self._flag_mngr

    @property
    def flag(self):
        """pd.Series: Corresponding data flag"""
        return self._flag_mngr.data

    def check_data(self, val):
        """Check data"""
        assert isinstance(val, pd.Series)

    def copy(self):
        """Copy method"""
        return deepcopy(self)

    def __len__(self):
        return len(self.data)

    def interpolate(self):
        """Interpolated method"""

        #TODO
        # Add automatic interpolation for polar coord (e.g. wind direction)
        # Check if this function must be improved/fixed
        interp_data = self.data.interpolate(method='index')

        # Set interp flag
        self.flag_mngr.set_bit_val('interp', True, index=self.data.isna())

        # Set data
        self.data = interp_data

    def get_flagged(self, flag_abbr):
        """Get flag value for given flag abbr"""
        return self.flag_mngr.get_bit_val(flag_abbr)


class TimeProfileManager(ProfileManger):
    """Time profile manager """

    @staticmethod
    def factory(data, event_mngr, index_lag=pd.Timedelta('0s'), u_u=None, u_r=None, u_s=None, u_t=None):
        """TimeProfileManager factory"""

        if len(data) == 0:
            return EmptyProfileManager(event_mngr)
        if (u_r is None) or (u_s is None) or (u_t is None):
            return TimeProfileManager(data, event_mngr, index_lag)
        if (u_s is None) or (u_t is None):
            raise NotImplementedError()
            #return TimeProfileErrTypeA(data, event_mngr, index_lag, err_r)
        else:
            raise NotImplementedError()
            #return TimeProfileErrTypeA(data, event_mngr, index_lag, err_r, err_s, err_t)

    def __init__(self, data, event_mngr, index_lag):
        """Constructor

        Args:
            data (pd.Series): pd.Series with index of type pd.TimedeltaIndex
            event_mngr (EventManager):
            index_lag (pd.Timedelta):

        """
        super().__init__(data, event_mngr)

        # Test
        assert isinstance(index_lag, pd.Timedelta)

        # Init attributes
        self._index_lag = index_lag

        # Reset index
        self._reset_index()

        # Set raw NA
        self._flag_mngr.set_bit_val('raw_na', True, self.data.isna())

    @property
    def index_lag(self):
        """pd.Timedelta: Index time lag"""
        return self._index_lag

    def check_data(self, val):
        """Overwrite check data"""
        super().check_data(val)

        # Test
        assert isinstance(val.index, pd.TimedeltaIndex)

    def resample(self, interval='1s', method='mean'):
        """Resample method

        Args:
            interval (str, optional): Resample interval. Default is '1s'.
            method (str, optional): Resample method, 'mean' (default) | 'sum'

        """

        resampler = self.data.resample(interval, label='right', closed='right')
        if method == 'mean':
            data_tmp = resampler.mean()
        elif method == 'sum':
            data_tmp = resampler.sum()
        else:
            raise AttributeError('Bad method value')

        # Reindex flag
        self.flag_mngr.reindex(data_tmp.index, 'resampled')

        # Set data
        self.data = data_tmp

    def _reset_index(self):
        """Set index start at 0s"""
        self._data.index = self.data.index - self.data.index[0]
        self._flag_mngr.data.index = self.flag.index - self.flag.index[0]

    def shift(self, periods):
        """

        Args:
          periods:

        Returns:


        """

        self._data = self.data.shift(periods)
        self._index_lag -= pd.Timedelta(periods, self.data.index.freq.name)


class TimeProfileErrTypeA(TimeProfileManager):
    """Error type A TimeProfileManager"""


class TimeProfileErrTypeB(TimeProfileManager):
    """Error type B TimeProfileManager"""


def load(search, prm_abbr, filter_empty=True):
    """Load parameter

    Args:
        search (str): Data loader search criterion
        prm_abbr (str): Positional parameter abbr
        filter_empty (bool): Filter empty data from search

    Returns:
        MultiTimeProfileManager

    .. uml::

        @startuml
        hide footbox

        load -> LocalDBLinker: load(search, prm_abbr)
        LocalDBLinker -> DatabaseManager: get_data(where=search, prm_abbr=prm_abbr)
        LocalDBLinker <- DatabaseManager : data
        load <- LocalDBLinker: data
        @enduml

    """

    # Init
    db_linker = LocalDBLinker()

    if filter_empty is True:
        search = "(" + search + ") & ~{_tag == '" + TAG_EMPTY_VAL + "'}"

    # Load data
    out = MultiTimeProfileManager()
    for data in db_linker.load(search, prm_abbr):
        out.append(
            TimeProfileManager.factory(
                data['data'],
                data['event'])
        )

    return out


def update_db(search, strict=False):
    """Update database.

    Args:
        search (str): prm_abbr search criteria.
        strict (bool, optional): If False, match for any sub-string.
            If True match for entire string. Default to False.

    .. uml::

        @startuml
        hide footbox

        update_db -> CSVHandler: handle(file_path, prm_abbr)
        activate CSVHandler

        CSVHandler -> GDPHandler: handle(file_path, prm_abbr)
        activate GDPHandler

        CSVHandler <- GDPHandler: data
        deactivate  GDPHandler

        update_db <- CSVHandler: data
        deactivate   CSVHandler

        @enduml


    """

    # Init linkers
    db_linker = LocalDBLinker()

    # Define chain of responsibility for loadgin from raw
    handler = CSVHandler()
    handler.set_next(GDPHandler())

    # Search prm_abbr
    if strict is True:
        search = {'where': Parameter.prm_abbr == search}
    else:
        search = {'where': Parameter.prm_abbr.contains(search)}

    prm_abbr_list = [
        arg[0] for arg in db_mngr.get_or_none(
            Parameter,
            search=search,
            attr=[[Parameter.prm_abbr.name]],
            get_first=False
        )
    ]

    # Log
    localdb.info(
        "Update db for following parameters: %s",
        prm_abbr_list
    )

    # Scan path
    origdata_path_scan = list(path_var.orig_data_path.rglob("*.*"))

    # Loop loading
    for prm_abbr in prm_abbr_list:

        # Log
        rawcsv.info("Start reading CSV files for '%s'", prm_abbr)

        # Scan files
        new_orig_data = []
        for file_path in origdata_path_scan:
            result = handler.handle(file_path, prm_abbr)
            if result:
                new_orig_data.append(result)

                # Log
                rawcsv.info(
                    "CSV files '%s' was treated", file_path
                )
            else:

                # Log
                rawcsv.debug(
                    "CSV files '%s' was left untouched", file_path
                )

        # Log
        rawcsv.info("Finish reading CSV files for '%s'", prm_abbr)
        rawcsv.info(
            "Found %d new data while reading CSV files for '%s'",
            len(new_orig_data),
            prm_abbr
        )

        # Log
        localdb.info(
            "Start inserting in local DB new found data for '%s'", prm_abbr
        )

        # Save to DB
        db_linker.save(new_orig_data)

        # Log
        localdb.info(
            "Finish inserting in local DB new found data for '%s'", prm_abbr
        )


class MultiProfileManager(list):
    """Mutli profile manager"""

    def load(self, *args, **kwargs):
        """Load method"""
        raise NotImplementedError('Please implement')


class MultiTimeProfileManager(MultiProfileManager):
    """Multi time profile manager"""

    def __init__(self, time_profiles_mngrs=tuple()):
        """

        Args:
            time_profiles_mngrs (iterable of TimeProfileManager):
        """
        super().__init__(time_profiles_mngrs)

    @staticmethod
    def load(search, prm_abbr):
        """Overwrite load method

        Args:
            search (str): Search criteria
            prm_abbr (str or list of str): Parameter to load

        """

        if isinstance(prm_abbr, str):
            prm_abbr = list(prm_abbr)

        for prm in prm_abbr:
            pass
            #data = load(search, prm)


    def map(self, func, inplace, *args, **kwargs):
        """Map individual TimeProfileManager"""
        if inplace:
            MultiTimeProfileManager(
                map(lambda x: func(x, *args, **kwargs), self)
            )
            out = None
        else:
            out = self.copy()
            MultiTimeProfileManager(
                map(lambda x: func(x, *args, **kwargs), out)
            )
        return out

    def copy(self):
        """Copy method"""
        return deepcopy(self)

    def append(self, value):
        """Overwrite of append method"""
        assert isinstance(value, (TimeProfileManager, EmptyProfileManager))
        super().append(value)

    def resample(self, interval='1s', method='mean', inplace=False):
        """Resample method"""
        return self.map(
            TimeProfileManager.resample, inplace=inplace,
            interval=interval, method=method
        )

    def interpolate(self, inplace=False):
        """Interpolate method"""
        return self.map(
            TimeProfileManager.interpolate, inplace=inplace
        )

    def synchronise(self, i_start=0, n_corr=300, window=30, i_ref=0):
        """Synchronise time profile

        Args:
          i_start (int): Start integer index. Default to 0.
          n_corr (int): Default to 300.
          window (int): Window size. Default to 30.
          i_ref (int): Reference TimeProfile index. Default to 0.

        Returns:


        """

        # Copy
        out = self.copy()

        # Select reference data
        ref_data = out[i_ref]

        # Find best corrcoef
        idx_offset = []
        window_values = range(-window, window)
        for test_data in out:
            corr_res = [
                crosscorr(
                    ref_data.data.iloc[i_start:i_start + n_corr],
                    test_data.data.iloc[i_start:i_start + n_corr],
                    lag=w)
                for w in window_values
            ]
            idx_offset.append(np.argmax(corr_res))

        for i, arg in enumerate(out):
            arg.shift(window_values[idx_offset[i]])
            arg.event_mngr.add_tag('sync')

        return out

    def plot(self, **kwargs):
        """Plot method"""

        basic_plot(self, **kwargs)
