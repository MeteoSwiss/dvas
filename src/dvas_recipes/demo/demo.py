"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: demonstration of recipe step function design
"""

# Relative imports from the dvas_recipe module
from .. import dynamic
from ..recipe import for_each_flight, for_each_var


@for_each_var
@for_each_flight
def sequential_dummy(dummy_arg=0):
    """ This is a simple function demonstrating how the @for_each_flight decorator can be used to
    run a recipe step sequentially on all the flights specified by the user.

    The secret lies in using dynamic.CURRENT_FLIGHT to access the flight_id, event_id and rig_id -
    the decorator takes care of looping these externally.

    Functions used for recipe steps should only take keyword arguments as input, to be defined
    accordingly in the recipe YAML file.

    Args:
       dummy_arg (int, optional): some dummy keyword argument for the example. It needs to be set
       accordingly in the recipe file.

    """

    # The event_id and rig_id can be easily extracted from the dynamic module inside dvas_recipes.
    (fid, eid, rid) = dynamic.CURRENT_FLIGHT

    print(f'dummy_arg: {dummy_arg}, fid: {fid}, eid: {eid}, rid: {rid}')
