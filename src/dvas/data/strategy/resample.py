"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Resample strategies

"""

# Import from external packages
from abc import ABCMeta, abstractmethod

# Import from current package
from .data import RSProfile


class ResampleDataStrategy(metaclass=ABCMeta):
    """Abstract class to manage data resample strategy.

    Resample strategy goal's is to obtain an homogeneous
    index, i.e constant interval and same for all profiles.
    """

    @abstractmethod
    def resample(self, *args, **kwargs):
        """Strategy required method"""


class ResampleRSDataStrategy(ResampleDataStrategy):
    """Class to manage resample of time data parameter"""

    def resample(self, data, interval='1s', method='mean'):
        """Implementation of resample method

        Args:
            data (dict): Dict or list of RSProfile
            interval (str, optional): Resample interval. Default is '1s'.
            method (str, optional): Resample method, 'mean' (default) | 'sum'

        """

        # TODO: THIS NEEDS TO BE FIXED !!!

        """
        for key, val in data.items():
            for i, arg in enumerate(val):

                resampler = arg.data.resample(
                    interval, label='right', closed='right'
                )

                if method == 'mean':
                    res = resampler.mean()

                elif method == 'sum':
                    res = resampler.sum()

                res.index -= res.index[0]
                val[i] = RSProfile(arg.event_mngr, data=res)

            data.update({key: val})

        return data
        """
