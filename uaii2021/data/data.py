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

from ..config.config import IdentifierManager, ConfigManagerMeta
from ..config.config import RawData
from .linker import LocalDBLinker, CSVLinker, ManufacturerCSVLinker


class TimeSeriesManager:

    def __init__(self, data, id_source, index_lag=pd.Timedelta('0s')):
        """

        Parameters
        ----------
        data: pd.Series with index of type pd.TimedeltaIndex
        id_source
        index_lag: pd.Timedelta
        """
        # Test
        assert isinstance(data, pd.Series)
        assert isinstance(data.index, pd.TimedeltaIndex)
        assert isinstance(index_lag, pd.Timedelta)

        # Init properties
        self._id_mngr = IdentifierManager(['ms', 'param', 'qc_status'])
        self._index_lag = index_lag

        data.name = self.id_mngr.join(id_source)
        data.index.name = None
        self._data = data

    @property
    def id_mngr(self):
        return self._id_mngr

    @property
    def index_lag(self):
        return self._index_lag

    @property
    def data(self):
        return self._data

    @staticmethod
    def load(id_source, load_order):
        for data_linker in load_order:
            data = data_linker.load(id_source)
            if data is not None:
                return TimeSeriesManager(data, id_source)

    def save(self, linker):
        linker.save(self.data.name, self.data)

    def interpolate(self):
        #TODO
        # Add interpolate of wind direction
        self._data = self.data.interpolate(method='index')
        return self

    def resample(self, interval='1s'):
        """

        Parameters
        ----------
        interval: str
            Resample interval

        Returns
        -------
        TimeSeriesManager

        """
        #TODO
        # Add resample of wind direction
        # Add sum function
        out = self.data.resample(interval, label='right', closed='right').mean()

        # Interpolate if oversampling
        if out.isna().any():
            out = out.interpolate(method='index')

        self._data = out
        return self

    def reset_index(self):
        """Set index start at 0s
        """
        self._data = self.data.reindex(self.data.index - self.data.index[0])
        return self

    def shift(self, periods):
        self._index_lag -= pd.Timedelta(periods, self.data.index.freq.name)
        self._data = self.data.shift(periods)
        return self

    def layering(self):
        pass

    def change_index_unit(self, new_unit: str):
        pass

    @staticmethod
    def synchronise(profile_series, i_start=0, n_corr=300, window=30, i_ref=0):
        """

        Parameters
        ----------
        profile_series: list of TimeSeriesManager

        Returns
        -------

        """

        # Copy
        profile_series = copy.deepcopy(profile_series)

        # Interpolate and resample
        for i, pf in enumerate(profile_series):
            pf.interpolate().resample().reset_index()

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

        return [arg.shift(window_values[idx_offset[i]]) for i, arg in enumerate(profile_series)]

    @staticmethod
    def crosscorr(datax, datay, lag=0, wrap=False, method='kendall'):
        """ Lag-N cross correlation.
        Shifted data filled with NaNs

        Parameters
        ----------
        lag : int, default 0
        datax, datay : pandas.Series objects of equal length
        method: str
            kendall, pearson, spearman

        Returns
        ----------
        crosscorr : float
        """
        if wrap:
            shiftedy = datay.shift(lag)
            shiftedy.iloc[:lag] = datay.iloc[-lag:].values
            return datax.corr(shiftedy, method=method)
        else:
            return datax.corr(datay.shift(lag), method=method)

class TimeDataFrameManager:

    def __init__(self, data):

        # Test
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.index, pd.TimedeltaIndex)

        self._data = data

    @property
    def data(self):
        return self._data



    @staticmethod
    def concat(profile_series):
        """

        Parameters
        ----------
        profile_series: list of TimeSeriesManger

        Returns
        -------

        """

        return TimeDataFrameManager(pd.concat([pf.data for pf in profile_series], axis=1))

    @staticmethod
    def load():
        #TODO
        # Method to load multiple data Series of same flight and same parameter
        pass

    def save(self, linker):
        for _, data in self.iteritems():
            data.save(linker)
