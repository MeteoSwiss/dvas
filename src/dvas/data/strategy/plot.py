"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Plotting strategies

"""

# Import from current package
from .data import MPStrategyAC
from ...plots import plots as dpp


class PlotStrategy(MPStrategyAC):
    """ Base class to manage the data plotting strategy for the MultiProfile class. """

    def execute(self, prfs, **kwargs):
        """ Call the proper plotting method for this strategy.

        Args:
            prfs (dict of Profile or RSProfile or GDPprofile): data to plot
            **kwargs: Keyword arguments to be passed down to the plotting function.

        """

        dpp.multiprf(prfs, **kwargs)


class RSPlotStrategy(MPStrategyAC):
    """ Child class to manage the plotting strategy for the MultiRSProfile class. """

    def execute(self, prfs, **kwargs):
        """ Call the proper plotting method for this strategy.

        Args:
            prfs (dict of Profile or RSProfile or GDPprofile): data to plot
            **kwargs: Keyword arguments to be passed down to the plotting function.

        """

        dpp.multiprf(prfs, **kwargs)


class GDPPlotStrategy(MPStrategyAC):
    """ Child class to manage the plotting strategy for the MultiGDPProfile class. """

    def execute(self, prfs, **kwargs):
        """ Call the proper plotting method for this strategy.

        Args:
            prfs (dict of Profile or RSProfile or GDPprofile): data to plot
            **kwargs: Keyword arguments to be passed down to the plotting function.

        """

        dpp.multiprf(prfs, **kwargs)
