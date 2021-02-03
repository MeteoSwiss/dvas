# -*- coding: utf-8 -*-
"""

Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains the high-level dvas entry points.

"""

import argparse
from pathlib import Path

from dvas import VERSION
from .hardcoded import DVAS_RECIPES
from .initialize import init_arena

def dvas_init():
    ''' The dvas_init entry point, wrapping around the actual init_arena function.
    '''

    # Use argparse to make dvas user friendly
    parser = argparse.ArgumentParser(description=
                                     'DVAS {}'.format(VERSION) +
                                     ' - Data Visualization and Analysis Software:' +
                                     ' Initialization function.',
                                     epilog='For more info: https://MeteoSwiss.github.io/dvas\n ',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('recipe', action='store', default='demo',
                        choices=DVAS_RECIPES,
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

# Make everything above actually work.
#if __name__ == "__main__":
#    main()
