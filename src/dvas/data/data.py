"""Module containing class and function for data management

"""

from abc import ABC, abstractmethod
from operator import itemgetter
import copy
import numpy as np
import pandas as pd
from enum import Enum, unique

from dataclasses import dataclass, field
from typing import List

from ..config.config import ConfigManagerMeta
from ..config.config import OrigData
from .linker import LocalDBLinker, CSVOutputLinker, OriginalCSVLinker

# Define
FLAG = 'flag'
VALUE = 'value'


@unique
class Flag(Enum):
    """ """
    RAW = 0
    RAW_NAN = 1
    INTERPOLATED = 2
    SYNCHRONIZED = 3


class TimeSeriesManager:
    """ """

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
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.index, pd.TimedeltaIndex)
        assert isinstance(index_lag, pd.Timedelta)

        # Set flag
        if all(data.columns == self._COL_NAME):
            pass
        elif all(data.columns == VALUE):
            self._set_flag(Flag.RAW.value)
            self._set_flag(Flag.RAW_NAN.value, np.isnan)
        else:
            raise AttributeError('Bad Dataframe format')

        # Init properties
        self._event_mngr = event_mngr
        self._index_lag = index_lag
        self._data = data

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

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load(search):
        """

        Args:
            search (str): Data loader search criterion

        Returns:


        """

        # Init
        db_linker = LocalDBLinker()

        # Load data
        out = []
        for data in db_linker.load(search):
            out.append(
                TimeSeriesManager(
                    data['data'],
                    data['event'])
            )

        return out

    def update_db(self):
        """

        Returns:

        """

        db_linker = LocalDBLinker()
        orig_data_linker = OriginalCSVLinker()

        orig_data_linker.load()

        db_linker.save()

    def _set_flag(self, bit_val, data_fct=None):
        """

        Args:
          bit_val:
          data_fct:  (Default value = None)

        Returns:


        """
        if data_fct is None:
            idx_data = np.ndarray(len(self), bool)
            idx_data.fill(True)
        else:
            idx_data = data_fct(self._data[VALUE])

        self._data.loc[idx_data, FLAG] = np.bitwise_or(
            self._data.loc[idx_data, FLAG],
            1 << bit_val)

    def _reset_flag(self, bit_val, data_fct=None):
        """

        Args:
          bit_val:
          data_fct:  (Default value = None)

        Returns:


        """
        if data_fct is None:
            idx_data = np.ndarray(len(self), bool)
            idx_data.fill(True)
        else:
            idx_data = data_fct(self._data[VALUE])

        self._data.loc[idx_data, FLAG] = np.bitwise_and(
                self._data.loc[idx_data, FLAG],
                np.bitwise_not(1 << bit_val))

    def _get_flag(self, bit_val):
        """

        Args:
          bit_val:

        Returns:


        """
        return np.bitwise_and(
            self._data.loc[FLAG],
            1 << bit_val)

    def interpolate(self):
        """ """
        #TODO
        # Add interpolate of wind direction
        self._set_flag(Flag.INTERPOLATED.value, np.isnan)
        self._data = self.data.interpolate(method='index')
        return self

    def _resample(self, interval='1s'):
        """

        Args:
          interval(str, optional): Resample interval (Default value = '1s')

        Returns:


        """
        #TODO
        # Add resample of wind direction
        # Add sum function
        out = self.data._resample(interval, label='right', closed='right').mean()

        # Interpolate if oversampling
        if out.isna().any():
            out = out.interpolate(method='index')

        self._data = out
        return self

    def _reset_index(self):
        """Set index start at 0s"""
        self._data = self.data.reindex(self.data.index - self.data.index[0])
        self._set_flag()
        return self

    def _shift(self, periods):
        """

        Args:
          periods:

        Returns:


        """
        self._index_lag -= pd.Timedelta(periods, self.data.index.freq.name)
        self._data = self.data.shift(periods)
        return self

    def layering(self):
        """ """
        pass

    def change_index_unit(self, new_unit: str):
        """

        Args:
          new_unit: str:

        Returns:


        """
        pass

    @staticmethod
    def synchronise(profile_series, i_start=0, n_corr=300, window=30, i_ref=0):
        """

        Args:
          profile_series(list of TimeSeriesManager):
          i_start:  (Default value = 0)
          n_corr:  (Default value = 300)
          window:  (Default value = 30)
          i_ref:  (Default value = 0)

        Returns:


        """

        # Copy
        profile_series = copy.deepcopy(profile_series)

        # Interpolate and resample
        for i, pf in enumerate(profile_series):
            pf.interpolate()._resample()._reset_index()

        ref_data = profile_series[i_ref]

        idx_offset = []
        window_values = range(-window, window)
        for test_data in profile_series:
            corr_res = [
                TimeSeriesManager.crosscorr(
                    ref_data.data.iloc[i_start:i_start+n_corr],
                    test_data.data.iloc[i_start:i_start+n_corr],
                    lag=w)
                for w in window_values
            ]
            idx_offset.append(np.argmax(corr_res))

        for i, arg in enumerate(profile_series):
            arg._shift(window_values[idx_offset[i]])
            arg.set_flag(Flag.SYNCHRONIZED.value)

        return

    @staticmethod
    def crosscorr(datax, datay, lag=0, wrap=False, method='kendall'):
        """Lag-N cross correlation.
        Shifted data filled with NaNs

        Args:
          lag(int, default 0, optional):  (Default value = 0)
          datax, datay(pandas.Series objects of equal length):
          method(str, optional): kendall, pearson, spearman (Default value = 'kendall')
          datax:
          datay:
          wrap:  (Default value = False)

        Returns:


        """
        if wrap:
            shiftedy = datay.shift(lag)
            shiftedy.iloc[:lag] = datay.iloc[-lag:].values
            return datax.corr(shiftedy, method=method)
        else:
            return datax.corr(datay.shift(lag), method=method)


class TimeDataFrameManager:
    """ """

    def __init__(self, data):

        # Test
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.index, pd.TimedeltaIndex)

        self._data = data

    @property
    def data(self):
        """ """
        return self._data

    @staticmethod
    def concat(profile_series):
        """

        Args:
          profile_series(list of TimeSeriesManger):

        Returns:


        """

        return TimeDataFrameManager(pd.concat([pf.data for pf in profile_series], axis=1))

    @staticmethod
    def load():
        """ """
        #TODO
        # Method to load multiple data Series of same flight and same parameter
        pass

    def save(self, linker):
        """

        Args:
          linker:

        Returns:


        """
        for _, data in self.iteritems():
            data.save(linker)
