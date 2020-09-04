"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Resample strategy

"""

# Import from external packages
from abc import ABCMeta, abstractmethod

# Import from current package


class ResampleDataStrategy(metaclass=ABCMeta):
    """Abstract class to manage data resample strategy"""

    @abstractmethod
    def resample(self, *args, **kwargs):
        """Strategy required method"""


class TimeResampleDataStrategy(ResampleDataStrategy):
    """Class to manage resample of time data parameter"""

    def resample(self, data, interval='1s', method='mean'):
        """Implementation of resample method

        Args:
            data (dict): Dict of list of ProfileManager
            interval (str, optional): Resample interval. Default is '1s'.
            method (str, optional): Resample method, 'mean' (default) | 'sum'

        """

        for key, val in data.items():
            for i, arg in enumerate(val):
                resampler = arg.data.resample(
                    interval, label='right', closed='right'
                )

                if method == 'mean':
                    val[i].data = resampler.mean()

                elif method == 'sum':
                    val[i].data = resampler.sum()

                val[i].data.index -= val[i].data.index[0]

            data.update({key: val})

        return data
