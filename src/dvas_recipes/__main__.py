# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains the high-level dvas entry points.

"""

import argparse
from pathlib import Path

from dvas import VERSION
from .hardcoded import DVAS_RECIPE_NAMES
from .hl_commands import init_arena, run_recipe

def dvas_init_arena():
    """ The dvas_init_arena entry point, wrapping around the actual init_arena function. """

    # Use argparse to make dvas user friendly
    parser = argparse.ArgumentParser(description=
                                     'DVAS {}'.format(VERSION) +
                                     ' - Data Visualization and Analysis Software:' +
                                     ' Initialization entry point.',
                                     epilog='For more info: https://MeteoSwiss.github.io/dvas\n ',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('recipe', action='store', default='demo',
                        choices=DVAS_RECIPE_NAMES,
                        help='The processing recipe to initialize.')

    parser.add_argument('-ap', '--arena_path', action='store', default=None,
                        metavar='./a/new/folder/',
                        help='Relative path & name for the new processing arena.')

    # Done getting ready. Now start doing stuff.
    # What did the user type in ?
    args = parser.parse_args()

    # If a path was specified, let's deal with it.
    if args.arena_path is not None:
        args.arena_path = Path(args.arena_path)

    # Launch the initialization of a new processing arena
    init_arena(args.recipe, arena_path=args.arena_path)

def dvas_run_recipe():
    """ The dvas_run_recipe entry point, wrapping around the actual run_recipe function. """

    # Use argparse to make dvas user friendly
    parser = argparse.ArgumentParser(description=
                                     'DVAS {}'.format(VERSION) +
                                     ' - Data Visualization and Analysis Software:' +
                                     ' Recipe initialization & execution entry point.',
                                     epilog='For more info: https://MeteoSwiss.github.io/dvas\n ',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('rcp_fn', action='store',
                        help=' (Path +) Name of the dvas recipe file (.rcp) to use.')

    # Done getting ready.
    # What did we get from the user ?
    args = parser.parse_args()

    # Feed this to the actual recipe routine
    run_recipe(Path(args.rcp_fn))
