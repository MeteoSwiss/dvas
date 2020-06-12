"""
Module containing class and function for data management

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .linker import LocalDBLinker, OriginalCSVLinker
from .math import crosscorr
from ..database.database import db_mngr
from ..database.model import Flag, Parameter
from ..database.model import EventsInfo, OrgiDataInfo
from ..database.model import Instrument
from ..database.database import OneDimArrayConfigLinker
from ..config.definitions.flag import RAWNA_ABBR, RESMPL_ABBR, UPSMPL_ABBR
from ..config.definitions.flag import INTERP_ABBR, SYNC_ABBR
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
        """Constructor

        Args:
            index (pd.Index): Pandas index
        """
        self._flags = {
            arg[Flag.flag_abbr.name]: arg
            for arg in db_mngr.get_flags()
        }

        self._data = pd.Series(
            np.zeros(len(index),), index=index, dtype=int
        )

    @property
    def flags(self):
        """dict: Flag abbr, description and bit position."""
        return self._flags

    @property
    def data(self):
        """pd.Series: Data corresponding flag value."""
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def get_bit_number(self, abbr):
        """Get bit number corresponding to given flag abbr"""
        return self.flags[abbr][self._FLAG_BIT_NM]

    def set_bit_val(self, abbr, index=None, set_val=True):
        """Set data flag value to one

        Args:
            abbr (str):
            index (pd.Index, optional): Default to None.
            set_val (bool, optional): Default to True

        """
        if index is None:
            index = self.data.index

        if set_val is True:
            self._data.loc[index] = self.data.loc[index].apply(
                lambda x: x | (1 << self.get_bit_number(abbr))
            )
        else:
            self._data.loc[index] = self.data.loc[index].apply(
                lambda x: x & ~(1 << self.get_bit_number(abbr))
            )

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
        assert isinstance(data, pd.Series)

        # Set attributes
        self._data = data
        self._data.name = None
        self._flag_mngr = FlagManager(data.index)
        self._event_mngr = event_mngr

    @property
    def data(self):
        """pd.Series: Data"""
        return self._data

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

    def copy(self):
        """Copy method"""
        return deepcopy(self)

    def __len__(self):
        return len(self.data)

    def interpolate(self):
        """Interpolated method"""

        # Test if resampled data
        assert self._flag_mngr.get_bit_val(RESMPL_ABBR).all(), (
            'Data have not been resampled'
        )

        #TODO
        # Add automatic interpolation for polar coord (e.g. wind direction)
        # Check if this function must be improved/fixed
        self._data = self.data.interpolate(method='index')

        # Set flag
        self._flag_mngr.set_bit_val(INTERP_ABBR)

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
        assert isinstance(data.index, pd.TimedeltaIndex)
        assert isinstance(index_lag, pd.Timedelta)

        # Init attributes
        self._index_lag = index_lag

        # Reset index
        self._reset_index()

        # Set raw NA
        self._flag_mngr.set_bit_val(RAWNA_ABBR, self.data.isna())

    @property
    def index_lag(self):
        """pd.Timedelta: Index time lag"""
        return self._index_lag

    def resample(self, interval='1s', method='mean'):
        """Resample method

        Args:
            interval (str, optional): Resample interval. Default is '1s'.
            method (str, optional): Resample method, 'mean' (default) | 'sum'

        """

        resampler = self.data.resample(interval, label='right', closed='right')
        if method == 'mean':
            self._data = resampler.mean()
        elif method == 'sum':
            self._data = resampler.sum()
        else:
            raise AttributeError('Bad method value')

        # Set resample flag
        self._flag_mngr.set_bit_val(RESMPL_ABBR)

        # Set up sampled flag
        self._flag_mngr.set_bit_val(
            UPSMPL_ABBR,
            self.data.isna() & (self._flag_mngr.get_bit_val(RAWNA_ABBR) != 1))

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

        # Check synchonised flag
        assert all(
            [arg.flag_mngr.get_bit_val(INTERP_ABBR).all() for arg in self]
        ), 'Please interpolate data before'

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
            arg.flag_mngr.set_bit_val(SYNC_ABBR)

        return out

    def plot(self):
        """Plot method"""

        plt.figure()
        for arg in self:
            plt.plot(arg.data)
