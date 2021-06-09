"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: demonstration of recipe step function design
"""

# Relative imports from the dvas_recipe module
from ... import dynamic
from ...recipe import for_each_flight, for_each_var

@for_each_var
@for_each_flight
def sequential_dummy(dummy_arg=0):
    """ This is a simple function demonstrating how the @for_each_flight decorator can be used to
    run a recipe step sequentially on all the flights specified by the user.

    The secret lies in using dynamic.THIS_FLIGHT to access the event_id and rig_id - the decorator
    takes care of looping these externally.

    Functions used for recipe steps should only take keyword arguments as input, to be defined
    accordingly in the recipe YAML file.

    Args:
       a(int, optional): some dummy keyword arguments, that needs to be set accordingly in the
           recipe file.

    """

    print('dummy_arg: {}, THIS_VAR: {}, THIS_FLIGHT: {}'.format(dummy_arg, dynamic.THIS_VAR,
          dynamic.THIS_FLIGHT))
