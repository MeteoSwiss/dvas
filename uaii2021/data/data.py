"""Module containing class and function for data management

"""

from abc import ABC, abstractmethod
from operator import itemgetter
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import List

from ..config.config import IdentifierManager, ConfigManagerMeta
from ..config.config import RawData
from .linker import LocalDBLinker, CSVLinker, ManufacturerCSVLinker


class TimeSeriesManager(pd.Series):

    _metadata = ['_load_order', '_offset']

    @abstractmethod
    def __init__(self, load_order, offset=0):
        """

        Parameters
        ----------
        id_source: list of str | str
        load_order: list of DataLinker instances

        """

        super().__init__()
        self._load_order = load_order
        self._offset = offset

    @property
    def _constructor(self):
        return TimeSeriesManager

    @property
    def id_source(self):
        return self.name

    @property
    def load_order(self):
        return self._load_order

    @property
    def offset(self):
        return self._offset

    def load(self):
        for data_linker in self.load_order:
            data = data_linker.load(self.id_source)
            if data is not None:
                assert isinstance(data, pd.Series)
                assert isinstance(data.index, pd.TimedeltaIndex)
                assert data.index[0] == pd.Timedelta('0s')
                super().__init__(data)
                break

    def save(self):
        LocalDBLinker.save(self.id_source, self.data)

    def shift(self, lag):
        """

        Parameters
        ----------
        lag: int
            Periods shift

        """
        super().__init__(self.data.shift(period=lag))
        self._offset += lag

    def interpolate(self):
        #TODO
        # Add interpolate of wind direct
        super().interpolate(method='index', inplace=True)

    def resample(self, interval='1s'):
        """

        Parameters
        ----------
        interval: str
            Resample interval

        Returns
        -------

        """
        #TODO
        # Add resample of wind direction
        # Add sum function
        self.data = self.data.resample(interval, label='right', closed='right').mean()

        # Interpolate if oversampling
        if self.data.isna().any():
            self.interpolate()

    def layering(self):
        pass

    def change_index_unit(self, new_unit: str):
        pass


class TimeDataFrameManager:

    def __init__(self, profile_series):
        """

        Parameters
        ----------
        profile_series: list of ProfileSeries
        """

        self._id_sources = [[arg.id_source for arg in profile_series]]
        self._data = pd.concat(
            [arg.data for arg in profile_series],
            join='inner',
            names=range(len(profile_series)))

    @property
    def id_sources(self):
        return self._id_sources

    @property
    def data(self):
        return self._data

    @staticmethod
    def synchronise(profile_series, i_start=0, n_corr=300, window=30, i_ref=0):
        """

        Parameters
        ----------
        profile_series: list of ProfileSeries

        Returns
        -------

        """

        # Interpolate and resample
        for pf in profile_series:
            pf.interpolate()
            pf.resample()

        ref_data = profile_series[i_ref]

        idx_offset = []
        window_values = range(-window, window)
        for test_data in profile_series:
            corr_res = [
                TimeDataFrameManager.crosscorr(
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