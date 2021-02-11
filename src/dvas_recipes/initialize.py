"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

from datetime import datetime
from pathlib import Path
import shutil

from .errors import DvasRecipesError
from .__init__ import recipe_path

def default_arena_path(recipe):
    """ Returns the default **relative** location and name of the dvas processing arena for a
    specific recipe: './dvas_recipe_arena/'

    Args:
        recipe (str): recipe name

    Returns:
        pathlib.Path: the default relative Path.
    """

    return Path('.', 'dvas_{}_arena'.format(recipe))

def init_arena(recipe, arena_path=None):
    ''' Initializes a new dvas prcoessing arena, based on a specific processing recipe.

    Args:
        recipe (str): the processing recipe. Must be one of ['demo', 'uaii22'].
        arena_path (pathlib.Path, optional): relative Path to the processing arena to initialize.
            Defaults to `./dvas_recipe_arena`.

    '''

    # Keep track of time
    start_time = datetime.now()

    # Some sanity checks to start with
    if arena_path is None:
        arena_path = default_arena_path(recipe)

    if not isinstance(arena_path, Path):
        raise DvasRecipesError(" Huh ! arena_path should be of type pathlib.Path, not: " +
                               "{}".format(type(arena_path)))

    print("Initializing a new dvas '{}' processing arena under {} ...".format(recipe, arena_path))

    # Require a new folder to avoid issues ...
    while arena_path.exists():
        arena_path = input('{} already exists.'.format(arena_path) +
                           ' Please specify a new (relative) path for the dvas arena:')
        arena_path = Path(arena_path)

    # Very well, setup the suitable directory
    shutil.copytree(recipe_path / recipe / "proc_arena", arena_path, ignore=None, dirs_exist_ok=False)

    # Say goodbye ...
    print('All done in %i s.' % ((datetime.now()-start_time).total_seconds()))
