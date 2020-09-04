"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Interpolate strategy

"""

# Import from external packages
from abc import ABCMeta, abstractmethod

# Import from current package


class InterpolateStrategy(metaclass=ABCMeta):
    """Abstract class to manage data interpolation strategy"""

    @ abstractmethod
    def interpolate(self, *args, **kwargs):
        """Strategy required method"""
