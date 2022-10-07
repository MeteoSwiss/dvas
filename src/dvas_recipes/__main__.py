# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains the high-level dvas entry points.

"""

import argparse
from pathlib import Path

from dvas import VERSION
from .hl_commands import init_arena, run_recipe, optimize


def dvas_init_arena():
    """ The dvas_init_arena entry point, wrapping around the actual init_arena function. """

    # Use argparse to make dvas user friendly
    parser = argparse.ArgumentParser(
        description='DVAS {}'.format(VERSION) + ' - Data Visualization and Analysis Software:' +
        ' Initialization entry point.',
        epilog='For more info: https://MeteoSwiss.github.io/dvas\n ',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--path', action='store', default=None,
                        metavar='./a/new/folder/',
                        help='Relative path & name for the new processing arena.')

    # Done getting ready. Now start doing stuff.
    # What did the user type in ?
    args = parser.parse_args()

    # If a path was specified, let's deal with it.
    if args.path is not None:
        args.path = Path(args.path)

    # Launch the initialization of a new processing arena
    init_arena(arena_path=args.path)


def dvas_optimize():
    """ The dvas_optimize entry point, wrapping around the optimize function designed to find the
    optimum chunk_size given a certain number of cpus (and memory).

    """

    # Use argparse to make dvas user friendly
    parser = argparse.ArgumentParser(
        description='DVAS {}'.format(VERSION) + ' - Data Visualization and Analysis Software:' +
        ' Optimization entry point.', epilog='For more info: https://MeteoSwiss.github.io/dvas\n ',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--n-cpus', action='store', default=None, type=int,
                        metavar='x',
                        help='Number of cpus used to run dvas. Defaults to None = max.')

    parser.add_argument('--prf-length', action='store', default=7001, type=int,
                        metavar='x',
                        help='Length of the test profiles. Defaults to 7001.')

    parser.add_argument('--chunk-min', action='store', default=50, type=int,
                        metavar='x',
                        help='Minimum chunk size to test. Defaults to 50.')

    parser.add_argument('--chunk-max', action='store', default=300, type=int,
                        metavar='x',
                        help='Maximum chunk size to test. Defaults to 300.')

    parser.add_argument('--n-chunk', action='store', default=6, type=int,
                        metavar='x',
                        help='Number of chunk samples to take. Defaults to 5.')

    args = parser.parse_args()

    optimize(n_cpus=args.n_cpus, prf_length=args.prf_length, chunk_min=args.chunk_min,
             chunk_max=args.chunk_max, n_chunk=args.n_chunk)


def dvas_run_recipe():
    """ The dvas_run_recipe entry point, wrapping around the actual run_recipe function. """

    # Use argparse to make dvas user friendly
    parser = argparse.ArgumentParser(
        description='DVAS {}'.format(VERSION) + ' - Data Visualization and Analysis Software:' +
        ' Recipe initialization & execution entry point.',
        epilog='For more info: https://MeteoSwiss.github.io/dvas\n ',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('rcp_fn', action='store', metavar='path/to/recipe.rcp',
                        help='(Path +) Name of the dvas recipe file (.rcp) to use.')

    parser.add_argument('fid_log_fn', action='store', metavar='path/to/eid_log.txt',
                        help='(Path +) Name of file linking F-ids to eids, rids, and edts.')

    parser.add_argument('-f', action='store', default=None, type=str,
                        help='Flight id(s) to treat specifically.', metavar='Fxy,Fxz,...')

    parser.add_argument('-s', action='store', default=None, type=str, metavar='00',
                        help='Skip recipe steps until this one. ' +
                        'If set, the DB reset will be force-disabled !')

    parser.add_argument('-e', action='store', default=None, type=str, metavar='99',
                        help='Skip recipe steps beyond this one. ')

    parser.add_argument('-d', action='store_true',
                        help='Force-set the logging level to DEBUG.')

    # Done getting ready.
    # What did we get from the user ?
    args = parser.parse_args()

    # Cleanup the inputs
    args.rcp_fn = Path(args.rcp_fn)
    args.fid_log_fn = Path(args.fid_log_fn)
    if args.f is not None:
        args.f = [item.strip(' ') for item in args.f.split(',')]

    # Feed this to the actual recipe routine
    run_recipe(args.rcp_fn, args.fid_log_fn, fid_to_treat=args.f,
               from_step_id=args.s, until_step_id=args.e, debug=args.d)
