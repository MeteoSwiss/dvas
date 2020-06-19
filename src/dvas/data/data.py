"""
Module containing class and function for data management

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import from external packages
from copy import deepcopy
import numpy as np
import pandas as pd

# Import from current package
from .linker import LocalDBLinker, OriginalCSVLinker
from ..plot.plot import basic_plot
from .math import crosscorr
from ..database.database import db_mngr
from ..database.model import Flag, Parameter
from ..database.model import EventsInfo, OrgiDataInfo
from ..database.model import Instrument
from ..database.database import OneDimArrayConfigLinker
from ..dvas_logger import localdb, rawcsv


# Define
FLAG = 'flag'
VALUE = 'value'
cfg_linker = OneDimArrayConfigLinker()


class FlagManager:
    """Flag manager class"""

    _FLAG_BIT_NM = Flag.bit_number.name
    _FLAG_ABBR_NM = Flag.flag_abbr.name
    _FLAG_DESC_NM = Flag.flag_desc.name

    def __init__(self, index):
        """
        Args:
            index (iterable): Flag index
        """
        self._flags = {
            arg[Flag.flag_abbr.name]: arg
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
        return self.flags[abbr][self._FLAG_BIT_NM]

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


class ProfileManger:
    """Profile manager"""

    def __init__(self, data, event_mngr):
        """Constructor

        Args:
            data (pd.Series): pd.Series with any index
            event_mngr (EventManager):

        """

        # Test
        self.check_data(data)

        # Set attributes
        self._data = data
        self._data.name = None
        self._flag_mngr = FlagManager(data.index)
        self._event_mngr = event_mngr

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
        self.event_mngr.rm_tag('raw')
        self.event_mngr.add_tag('derived')

    @property
    def event_mngr(self):
        """EventManager: Corresponding data event manager"""
        return self._event_mngr

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

    def __init__(self, data, event_mngr, index_lag=pd.Timedelta('0s')):
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


def load(search, prm_abbr):
    """

    Args:
        search (str): Data loader search criterion
        prm_abbr (str):

    Returns:


    """

    # Init
    db_linker = LocalDBLinker()

    # Load data
    out = MultiTimeProfileManager()
    for data in db_linker.load(search, prm_abbr):
        out.append(
            TimeProfileManager(
                data['data'],
                data['event'])
        )

    return out


def update_db(prm_contains):
    """Update database.

    Args:
        prm_contains (str): Parameter abbr search criteria. Use '%' for any
            character.

    """

    # Init linkers
    db_linker = LocalDBLinker()
    orig_data_linker = OriginalCSVLinker()

    # Search prm_abbr
    prm_abbr_list = [
        arg[0] for arg in db_mngr.get_or_none(
            Parameter,
            search={'where': Parameter.prm_abbr.contains(prm_contains)},
            attr=[[Parameter.prm_abbr.name]],
            get_first=False
        )
    ]

    localdb.info(
        "Update db for following parameters: %s",
        prm_abbr_list
    )

    # Loop loading
    for prm_abbr in prm_abbr_list:

        # Log
        rawcsv.info("Start reading CSV files for '%s'", prm_abbr)

        # Search exclude file names
        exclude_file_name = db_mngr.get_or_none(
            EventsInfo,
            search={
                'where': (
                    (Parameter.prm_abbr == prm_abbr) &
                    (Instrument.instr_id != '')
                ),
                'join_order': [Parameter, OrgiDataInfo, Instrument]},
            attr=[[EventsInfo.orig_data_info.name, OrgiDataInfo.source.name]],
            get_first=False
        )

        # Load
        new_orig_data = orig_data_linker.load(prm_abbr, exclude_file_name)

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


class MultiTimeProfileManager(list):
    """Multi time profile manager"""

    def __init__(self, time_profiles_mngrs=tuple()):
        """

        Args:
            time_profiles_mngrs (iterable of TimeProfileManager):
        """
        super().__init__(time_profiles_mngrs)

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
        assert isinstance(value, TimeProfileManager)
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
