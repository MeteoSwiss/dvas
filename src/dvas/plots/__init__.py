"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

# Import from this package
from .plots import *
from .utils import set_mplstyle

# Immediately enable the base look for DVAS plots, as soon as we load the module.
set_mplstyle(style='base')
