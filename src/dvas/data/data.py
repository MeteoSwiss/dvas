"""
Module containing class and function for data management


"""

from abc import ABC, abstractmethod
from operator import itemgetter
import copy
import numpy as np
import pandas as pd
from enum import Enum, unique
from functools import wraps
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import List

from ..config.config import ConfigManagerMeta
from ..config.config import OrigData
from ..config.definitions.flag import RAWNA_ABBR, RESMPL_ABBR, UPSMPL_ABBR
from ..config.definitions.flag import INTERP_ABBR, SYNC_ABBR, AUTOQC_ABBR
from .linker import LocalDBLinker, CSVOutputLinker, OriginalCSVLinker
from ..database.database import db_mngr
from ..database.model import Flag
from ..database.database import ConfigLinker
from .math import crosscorr

from mdtpyhelper.misc import timer

# Define
FLAG = 'flag'
VALUE = 'value'
cfg_linker = ConfigLinker()

class FlagManager:

    _FLAG_BIT_NM = Flag.bit_number.name
    _FLAG_ABBR_NM = Flag.flag_abbr.name
    _FLAG_DESC_NM = Flag.desc.name

    def __init__(self, index, ):

        self._flags = {
            arg[Flag.flag_abbr.name]: arg
            for arg in db_mngr.get_flags()
        }

        self._data = pd.Series(
            np.zeros(len(index),), index=index, dtype=int
        )

    @property
    def flags(self):
        return self._flags

    @property
    def data(self):
        return self._data

    def get_bit_number(self, abbr):
        return self.flags[abbr][self._FLAG_BIT_NM]

    def set_bit_val(self, abbr, index=None):
        if index is None:
            self._data = self.data.apply(
                lambda x: x | (1 << self.get_bit_number(abbr))
            )
        else:
            self._data.loc[index] = self.data.loc[index].apply(
                lambda x: x | (1 << self.get_bit_number(abbr))
            )

    def get_bit_val(self, abbr):
        return self.data.apply(
            lambda x: x & (1 << self.get_bit_number(abbr))
        )


class TimeProfileManager:
    """Time profile manager """

    _COL_NAME = [VALUE, FLAG]

    def __init__(self, data, event_mngr, index_lag=pd.Timedelta('0s')):
        """

        Parameters
        ----------
        data: pd.Series with index of type pd.TimedeltaIndex
        event_mngr
        index_lag: pd.Timedelta
        """

        # Test
        assert isinstance(data, pd.Series)
        assert isinstance(data.index, pd.TimedeltaIndex)
        assert isinstance(index_lag, pd.Timedelta)

        # Init attributes
        self._flag_mngr = FlagManager(data.index)
        self._event_mngr = event_mngr
        self._index_lag = index_lag
        self._data = data

        # Reset index
        self._reset_index()

        # Set raw NA
        self._flag_mngr.set_bit_val(RAWNA_ABBR, self.data.isna())

        # Set data attributes
        self._data.name = None

    @property
    def event_mngr(self):
        """ """
        return self._event_mngr

    @property
    def index_lag(self):
        """ """
        return self._index_lag

    @property
    def data(self):
        """ """
        return self._data

    @property
    def flag(self):
        return self._flag_mngr.data

    def __len__(self):
        return len(self.data)

    def _interpolate(self):
        """ """

        # Test if resampled data
        assert self._flag_mngr.get_bit_val(RESMPL_ABBR).all(), (
            'Data have not been resampled'
        )

        #TODO
        # Add automatique interpolation for polar coord (e.g. wind direction)
        self._data = self.data.interpolate(method='index')

        # Set flag
        self._flag_mngr.set_bit_val(INTERP_ABBR)

    def _resample(self, interval='1s', method='mean'):
        """

        Args:
            interval (str, optional): Resample interval. Default is '1s'.
            method (str, optional): Resample method, 'mean' (default) | 'sum'
            inplace (bool, optional): Default is True

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
        self._flag_mngr._data.index = self.flag.index - self.flag.index[0]

    def _shift(self, periods):
        """

        Args:
          periods:

        Returns:


        """

        self._data = self.data.shift(periods)
        self._index_lag -= pd.Timedelta(periods, self.data.index.freq.name)

    def change_index_unit(self, new_unit: str):
        """

        Args:
          new_unit: str:

        Returns:


        """
        pass

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


def update_db(prm_abbr):
    """

    Returns:

    """

    # Load CSV data
    db_linker = LocalDBLinker()
    orig_data_linker = OriginalCSVLinker()
    new_orig_data = orig_data_linker.load(prm_abbr)

    # Save to DB
    db_linker.save(new_orig_data)


class MultiTimeProfileManager(list):
    """ """

    @staticmethod
    def map_inplace(obj, func, map_inplace, *args, **kwargs):
        if map_inplace:
            MultiTimeProfileManager(
                map(lambda x: func(x, *args, **kwargs), obj)
            )
            return
        else:
            out = obj.copy()
            MultiTimeProfileManager(
                map(lambda x: func(x, *args, **kwargs), out)
            )
            return out

    def copy(self):
        """Overwrite of copy method"""
        return copy.deepcopy(self)

    def append(self, value):
        """Overwrite of append method"""
        assert isinstance(value, TimeProfileManager)
        super().append(value)

    def resample(self, interval='1s', method='mean', inplace=False):
        return self.map_inplace(
            self, TimeProfileManager._resample, map_inplace=inplace,
            interval=interval, method=method
        )

    def interpolate(self, inplace=False):
        return self.map_inplace(
            self, TimeProfileManager._interpolate, map_inplace=inplace
        )

    def synchronise(self, i_start=0, n_corr=300, window=30, i_ref=0):
        """

        Args:
          profile_series(list of TimeSeriesManager):
          i_start:  (Default value = 0)
          n_corr:  (Default value = 300)
          window:  (Default value = 30)
          i_ref:  (Default value = 0)

        Returns:


        """

        # Check synchonised flag
        assert all(
            [arg._flag_mngr.get_bit_val(INTERP_ABBR).all() for arg in self]
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
            arg._shift(window_values[idx_offset[i]])
            arg._flag_mngr.set_bit_val(SYNC_ABBR)

        return out

    @timer
    def plot(self):
        """ """

        fig = plt.figure()
        for arg in self:
            plt.plot(arg.data)
