"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This file contains the very-high-level dvas_recipe commands, to initialize and run them.
"""

import os
from datetime import datetime
from pathlib import Path
import shutil
import numpy as np

from .errors import DvasRecipesError
from .hardcoded import DVAS_RECIPE_FNEXT
from .utils import default_arena_path, recipe_storage_path
from .recipe import Recipe

def init_arena(arena_path=None):
    ''' Initializes a new dvas prcoessing arena.

    Args:
        arena_path (pathlib.Path, optional): relative Path to the processing arena to initialize.
            Defaults to None == ``./dvas_proc_arena/``.

    '''

    # Keep track of time
    start_time = datetime.now()

    # Some sanity checks to start with
    if arena_path is None:
        arena_path = default_arena_path()

    if not isinstance(arena_path, Path):
        raise DvasRecipesError(" Huh ! arena_path should be of type pathlib.Path, not: " +
                               "{}".format(type(arena_path)))

    print("Initializing a new dvas processing arena under {} ...".format(arena_path))

    # Require a new folder to avoid issues ...
    while arena_path.exists():
        arena_path = input('{} already exists.'.format(arena_path) +
                           ' Please specify a new (relative) path for the dvas arena:')
        arena_path = Path(arena_path)

    # Very well, setup the suitable directory
    shutil.copytree(recipe_storage_path() / "proc_arena",
                    arena_path, ignore=None, dirs_exist_ok=False)

    # Say goodbye ...
    print('All done in %i s.' % ((datetime.now()-start_time).total_seconds()))


def run_recipe(rcp_fn=None, flights=None):
    ''' Loads and execute a dvas recipe.

    Args:
        rcp_fn (pathlib.Path, optional): path to the specific dvas recipe to execute.
        flights (pathlib.Path, optional): path to the text file specifiying specific radiososnde
            flights to process. The file should contain one tuple of evt_id, rig_rid per line,
            e.g.::

                # This is a comment
                # Each line should contain the event_id, rig_id
                # These must be integers !
                12345, 1
                12346, 1

    '''

    # If warranted, look for local dvas recipe files.
    if rcp_fn is None:
        print("Looking for local dvas recipe (.rcp) files ...")
        rcp_fns = [item for item in os.listdir('.')
                   if item[:-len(DVAS_RECIPE_FNEXT)]==DVAS_RECIPE_FNEXT]

        if len(rcp_fns) == 0:
            raise DvasRecipesError(' Ouch ! No dvas recipe file (.rcp) found here !')
        if len(rcp_fns) > 1:
            print('Found {} dvas recipes: which one should be launched ?'.format(len(rcp_fns)))
            rcp_fn = None
            while rcp_fn not in rcp_fns:
                rcp_fn = input('Please specifiy:')
        else:
            rcp_fn = rcp_fns[0]

        # Let us not forget to turn the str into a Path
        rcp_fn = Path('.', rcp_fn)

    else:
        # Make sure the user specified info is valid.
        if not isinstance(rcp_fn, Path):
            raise DvasRecipesError('Ouch ! rcp_fn should be of type pathlib.Path, ' +
                                   'not: {}'.format(type(rcp_fn)))

        if not rcp_fn.exists():
            raise DvasRecipesError('Ouch ! {} does not exist.'.format(rcp_fn))

    # If warranted, extract the specific flights' eid/rid that should be processed
    if flights is not None:
        if not isinstance(flights, Path):
            raise DvasRecipesError('Ouch ! flights should be of type pathlib.Path, not: '+
                                   '{}'.format(type(flights)))
        flights = np.atleast_2d(np.genfromtxt(flights, comments='#', delimiter=',', dtype=int))


    # Very well, I am now ready to start initializing the recipe.
    rcp = Recipe(rcp_fn, flights=flights)

    starttime=datetime.now()

    # Launch the procesing !
    rcp.execute()

    print('\n\n All done - {} steps of "{}" recipe completed in {}s.'.format(rcp.n_steps, rcp.name,
        (datetime.now()-starttime).total_seconds()))
