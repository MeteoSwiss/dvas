"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains general dvas_recipe utilities.
"""

from pathlib import Path

def recipe_storage_path():
    """ Returns the absolute path to the stored dvas_recipes location, from where they can be
    copied locally.

    """
    return Path(__file__).resolve(strict=True).parent

def default_arena_path(recipe):
    """ Returns the default **relative** location and name of the dvas processing arena for a
    specific recipe: './dvas_recipe_arena/'

    Args:
        recipe (str): recipe name

    Returns:
        pathlib.Path: the default relative Path.
    """

    return Path('.', 'dvas-arena_{}'.format(recipe))
