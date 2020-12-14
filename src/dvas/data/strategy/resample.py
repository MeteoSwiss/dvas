"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Resample strategies

"""

# Import from external packages
from abc import ABCMeta, abstractmethod
from pandas.core.resample import TimedeltaIndexResampler

# Import from current package
from ...errors import dvasError


# TODO
#  Adapt this strategy to MultiIndex

class ResampleStrategyAbstract(metaclass=ABCMeta):
    """Abstract class to manage data resample strategy.

    Resample strategy goal's is to obtain an homogeneous
    index, i.e constant interval and same for all profiles.

    Resampling can only applied to radiosonde profile, i.e. with a
    datetime delta index.
    """

    @abstractmethod
    def resample(self, *args, **kwargs):
        """Strategy required method"""


class ResampleRSDataStrategy(ResampleStrategyAbstract):
    """Class to manage resample of time data parameter"""

    def resample(
            self, data, method='mean', rule='1s',
            closed='right', label='right', **kwargs
    ):
        """Implementation of resample method

        Args:
            data (list of Profile): Data to resample
            method (str, optional): Resample method, 'mean' (default) | 'sum' | 'max' | 'min'
            rule (str, optional): See pandas doc
            closed (str, optional): See pandas doc
            label (str, optional): See pandas doc
            **kwargs: Arbitrary keyword arguments

        Note:
            Resampling uses `pandas.DataFrame.resample https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html` method arguments"""

        # Define
        oper = {
            'mean': TimedeltaIndexResampler.mean,
            'sum': TimedeltaIndexResampler.sum,
            'max': TimedeltaIndexResampler.max,
            'min': TimedeltaIndexResampler.min,
        }

        data_rspl = []
        for arg in data:

            # Resampler
            rspler = arg.data.resample(
                rule=rule, closed=closed, label=label, **kwargs
            )

            # Set data
            try:
                data_rspl.append(
                    arg.__class__(
                        arg.info,
                        oper[method](rspler).reset_index()
                    )
                )
            except KeyError:
                raise dvasError(f"Unknown resample method '{method}'")

        return data_rspl
