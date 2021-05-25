"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas_recipes.utils module.

"""

from pathlib import Path

# Import module to test
from dvas_recipes import utils as dru

def test_initialize_recipe():
    """ Testing the recipe initialization routine. """

    this_file_path = Path(__file__).resolve()

    try:
        # This will fail, because the recipe configuratiomn file does not exist.
        # But I use the error to make sure I was looking for the correct file.
        # Adapted from the reply of boertel and Mingye Wang on StackOverflow:
        # https://stackoverflow.com/questions/40666924
        dru.initialize_recipe(this_file_path)
    except FileNotFoundError as not_found:
        assert Path(not_found.filename) == this_file_path.with_suffix('.yml')
