"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

# Import from Python packages
from pathlib import Path
import matplotlib.pyplot as plt

# Import from this package
from .plots import *

from ..dvas_environ import path_var as env_path_var
from .. import pkg_path
from . import utils as pu

# Immediately enable the base look for DVAS plots, as soon as we load the module.
plt.style.use(str(Path(pkg_path, 'plots', 'mpl_styles', pu.PLOT_STYLES['base'])))

# Uncomment the following for having the pretty LaTeX plots by default
#plt.style.use(str(Path(pkg_path, 'plot', 'mpl_styles', pu.PLOT_STYLES['latex'])))