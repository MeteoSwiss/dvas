"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Plotting strategies

"""

# Import from external packages
from abc import ABCMeta, abstractmethod

# Import from current package
from ...plot.plot import basic_plot, basic_alt_plot


class PlotStrategy(metaclass=ABCMeta):
    """Abstract class to manage data plotting strategy"""

    @abstractmethod
    def plot(self, *args, **kwargs):
        """Strategy required method"""


class TimePlotStrategy(PlotStrategy):
    """CLass to manage plotting of time data parameter"""

    def plot(self, values, event_mngr, **kwargs):
        """Plot time profiles"""

        basic_plot(values['data'], **kwargs)


class AltTimePlotStartegy(PlotStrategy):
    """Class to manage plotting of time data parameter"""

    def plot(self, values, event_mngr, *args, **kwargs):
        """PLot data in altitude"""

        values_new = []
        for arg_data, arg_alt in zip(values['data'], values['alt']):
            arg_data.index = arg_alt.values
            values_new.append(arg_data)

        basic_alt_plot(values_new, **kwargs)
