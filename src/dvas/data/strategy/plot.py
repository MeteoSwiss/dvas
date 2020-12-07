"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Plotting strategies

"""

# Import from external packages
from abc import ABCMeta#, abstractmethod

# Import from current package
from ...plots import plots as dpp

class PlotStrategy(metaclass=ABCMeta):
    """ Base class to manage the data plotting strategy for the MultiProfile class. """

    #TODO: interesting pylint suggestion here. It is suggesting to create a simple function
    # rather than a full class. Any merit to this suggestion ?
    def plot(self, prfs, **kwargs):
        """ Call the proper plotting method for this strategy.

        Args:
            prfs (dict of Profile or RSProfile or GDPprofile): data to plot
            **kwargs: Keyword arguments to be passed down to the plotting function.

        """

        dpp.multiprf(prfs, **kwargs)

class RSPlotStrategy(PlotStrategy):
    """ Child class to manage the plotting strategy for the MultiRSProfile class. """

    def plot(self, prfs, **kwargs):
        """ Call the proper plotting method for this strategy.

        Args:
            prfs (dict of Profile or RSProfile or GDPprofile): data to plot
            **kwargs: Keyword arguments to be passed down to the plotting function.

        """

        dpp.multiprf(prfs, **kwargs)

class GDPPlotStrategy(PlotStrategy):
    """ Child class to manage the plotting strategy for the MultiGDPProfile class. """

    def plot(self, prfs, **kwargs):
        """ Call the proper plotting method for this strategy.

        Args:
            prfs (dict of Profile or RSProfile or GDPprofile): data to plot
            **kwargs: Keyword arguments to be passed down to the plotting function.

        """

        dpp.multiprf(prfs, **kwargs)
