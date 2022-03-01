"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: basic high-level recipes for the UAII2022 campaign
"""

# Import general Python packages

# Import dvas modules and classes
from dvas.logger import recipes_logger as logger
from dvas.logger import log_func_call

# Import from dvas_recipes
# from .. import dynamic
from ..recipe import for_each_flight, for_each_var
# from ..errors import DvasRecipesError


@for_each_var
@for_each_flight
@log_func_call(logger, time_it=True)
def cleanup(dt, prec):
    """ Highest-level function responsible for doing an initial cleanup of the data.

    Args:
        dt (int|float): time step, in seconds, with which to resample all the RS profiles.
        prec (float): precision, in seconds, with which to round the time steps.

    """

    import pdb
    pdb.set_trace()
