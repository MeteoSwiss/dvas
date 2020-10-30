# -*- coding: utf-8 -*-
"""

Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains the high-level dvas entry points.

"""

import argparse
from pathlib import Path
from datetime import datetime
import shutil

from .dvas_version import VERSION
from . import expl_path

DEFAULT_ARENA_PATH = Path('.', 'dvas_arena')

def dvas_init():
    ''' The high-level initialization function, to setup a new DVAS processing arena'''

    start_time = datetime.now()
    print('')
    arena_path = DEFAULT_ARENA_PATH
    print('Initializing a new DVAS processing arena under "%s" ...' % (arena_path))

    # For now, require a new folder to avoid issues ...
    while arena_path.exists():
            arena_path = \
              input('%s already exists. Please specify a new (relative) path for the dvas arena:'
                    % (arena_path))
            arena_path = Path(arena_path)

    # Very well, setup the suitable directory
    #arena_path.mkdir()
    shutil.copytree(expl_path, arena_path, ignore=None, dirs_exist_ok=False)

    # Say goodbye ...
    print('All done in %i s.' % ((datetime.now()-start_time).total_seconds()))

def main():
    ''' The main function, offering the entry point to the underlying linestats routine.
    '''

    # Use argparse to make ldvas user friendly
    parser = argparse.ArgumentParser(description=
                                     'DVAS %s - Data Visualization and Analysis Software'
                                     % (VERSION),
                                     epilog='For more info: https://MeteoSwiss.github.io/dvas\n ',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--init', action='store_true',
                        help='Initialize a new DVAS processing arena.')

    # Done getting ready. Now start doing stuff.
    # What did the user type in ?
    args = parser.parse_args()

    # Give priority to the init function
    if args.init:
        # Launch the initialization of a new processing arena
        dvas_init()


# Make everything above actually work.
if __name__ == "__main__":
    main()
