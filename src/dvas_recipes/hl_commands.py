"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This file contains the very-high-level dvas_recipe commands, to initialize and run them.
"""

import multiprocessing as mpr
from datetime import datetime
import time
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs

from dvas.data.strategy.data import GDPProfile
from dvas.data.data import MultiGDPProfile
from dvas.database.database import InfoManager, DatabaseManager
from dvas.database.model import Model as TableModel
from dvas.database.model import Object as TableObject
from dvas.tools.gdps.gdps import combine
from dvas import dynamic as dyn
from dvas.environ import path_var
import dvas.plots.utils as dpu

from .errors import DvasRecipesError
from .utils import default_arena_path, demo_storage_path, recipe_storage_path
from .recipe import Recipe

def init_arena(arena_path=None):
    ''' Initializes a new dvas processing arena.

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

    # Very well, setup the config files for the dvas database initialization
    shutil.copytree(demo_storage_path(), arena_path, ignore=None, dirs_exist_ok=False)

    # And also copy the dvas recipes, in case the user wants to use these
    shutil.copytree(recipe_storage_path(), arena_path, ignore=None, dirs_exist_ok=True)

    # Say goodbye ...
    print('All done in %i s.' % ((datetime.now()-start_time).total_seconds()))

def optimize(n_cpus=None, prf_length=7001, chunk_min=50, chunk_max=300, n_chunk=6):
    """ Measures which chunk size provides the fastest computation time for the combine() function.

    This function is relatively dumb: it will try all requested chunk sizes, and measure which is
    the fastest.

    Warning:
        For best results, it is recommended to **not** run anything else in parallel to this test,
        in particular if all cpus are requested.

    Note:
        To reduce computing time, profiles get "sliced" when they are to be combined, in order to
        exploit, if available, multiple cpus. The size of the individual chunks is a parameter that
        can be adjusted manually by the user of dvas recipes. The best chunk size will be system
        dependant. A size too low will increase the computation time through additional loops.
        A size too large will drastically increase (and possible top-up) the RAM requirements, thus
        also increasing the computing time.

    Args:
        n_cpu (int, optional): number of cpus to use for the test. Defaults to None = all available.
        prf_length (int, optional): length of the test profiles. Default to 7001 elements.
        chunk_min (int, optional): minimum chunk size to test. Defaults to 50.
        chunk_max (int, optional): maximum chunk size to test. Defaults to 300.
        n_chunk (int, optional): number of chunk samples to take. Defaults to 6.

    """

    # Some sanity checks at first.
    if n_cpus is None or n_cpus > mpr.cpu_count():
        n_cpus = mpr.cpu_count()

    # Let's create DB in memory - a DB is required to run the combine function (which requires
    # access to the 'mid', 'oid', etc ... tables). Createing it in memory will avoid confusing the
    # user, and avoid issues when creating the actual one down the line.
    # The down side is that things are about to get a bit manual ...
    dyn.DB_IN_MEMORY = True
    print('\n Setting-up a temporary in-memory dvas database ...')

    # Set the config file path, so that we can have a DB initialize with proper parameters.
    # Point towards the core dvas file, so that this can be run from anywhere.
    setattr(path_var, 'config_dir_path', demo_storage_path() / 'config')

    # Actually create the database
    db_mngr = DatabaseManager()

    # Now, let's prepare some data for the DB tables ... We want to simulate 3 profiles from a
    # single flight.
    db_data = [{'mdl_name': 'RS41-GDP_001',
                'srn': 'prf_1',
                'pid': '1'},
               {'mdl_name': 'RS41-GDP_001',
                'srn': 'prf_2',
                'pid': '1'},
               {'mdl_name': 'RS41-GDP_001',
                'srn': 'prf_3',
                'pid': '1'},]

    # Next prepare these custom profiles. They require oids that need to be generated by the DB
    # tables themsleves. This is were things get super manual and self-referencing ...
    gdp_prfs = []
    for (ind, item) in enumerate(db_data):
        # Get the model thingy ...
        model = db_mngr.get_or_none(TableModel,
            search={'where': TableModel.mdl_name == item['mdl_name']})

        # ... and use it to create a dedicated instrument entry in the Object Table ...
        oid = TableObject.create(srn=item['srn'], pid=item['pid'], model=model).oid
        # ... so we can add it to the db_data we assembled above ...
        db_data[ind].update({'oid': oid})

        # Prepare some datasets to play with: first an InfoManager with the oid we just minted ...
        info = InfoManager('20210616T0000Z', oid, tags={'e:1', 'r:1'})

        # And then some random data ...
        data = pd.DataFrame({'alt': np.arange(0, prf_length, 1),
                               'tdt': np.arange(0, prf_length, 1),
                               'val': np.random.rand(prf_length),
                               'flg': None,
                               'ucr': np.random.rand(prf_length),
                               'ucs': np.random.rand(prf_length),
                               'uct': np.random.rand(prf_length),
                               'ucu': np.random.rand(prf_length)})

        # Let's assemble an actuale GDPProfile, and store it for later
        gdp_prfs += [GDPProfile(info, data)]

    # We can now build a bona fide MultiGDPProfile to test things out. Here again, we assume that
    # certain variable names have been properly defined in the db configuration files.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': 'temp', 'tdt': 'time', 'alt': 'gph', 'flg': None,
                     'ucr': 'temp_ucr', 'ucs': 'temp_ucs',
                     'uct': 'temp_uct', 'ucu': 'temp_ucu'},
                     gdp_prfs)

    print('\n Computing the weighted-mean of 3 profiles with different chunk sizes,'+
          f' with {n_cpus} cpus ...\n')

    # We are now ready to launch the combination routine, which relies on Profile slicing.
    run_times=[]
    # What chunk sizes need to be tested ?
    chunk_sizes = np.linspace(chunk_min, chunk_max, n_chunk, dtype=int)
    chunk_sizes = list(chunk_sizes) + list(chunk_sizes)[::-1]
    # Loop through all them them twice in reverse orders (just to get two data points for each).
    for chunk_size in chunk_sizes:
        start_time = time.perf_counter()
        # Here I don't actually care about the outcome, just the time it takes to get there ...
        _ = combine(multiprf, binning=1, method='weighted mean', chunk_size=chunk_size,
                    n_cpus=n_cpus)
        # Store this info, and also print it to screen ...
        run_times += [time.perf_counter() - start_time]
        print('  chunk_size: {} <=> {:.2f}s'.format(chunk_size, run_times[-1]))

    # Let's activate the default dvas plotting style. No LaTeX here.
    dpu.set_mplstyle('base')

    plt.figure(figsize=(dpu.WIDTH_ONECOL, 4.5))

    gs_info = gs.GridSpec(1, 1, height_ratios=[1], width_ratios=[1],
                     left=0.15, right=0.95, bottom=0.2, top=0.9, wspace=0.02, hspace=0.08)

    ax0 = plt.subplot(gs_info[0, 0])

    ax0.plot(chunk_sizes, run_times, 'kx')

    ax0.set_ylabel('Run time [s]', labelpad=10)
    ax0.set_xlabel('Chunk size')

    ax0.text(0.95, 0.05, f'n_cpus: {n_cpus}',
                 horizontalalignment='right', verticalalignment='bottom',
                 transform=ax0.transAxes)

    fn_out = 'dvas_optimize_{}_{}-cpus.pdf'.format(datetime.now().strftime('%Y%m%dT%H%M%S'),
                                                   n_cpus)

    plt.savefig(fn_out)

    print(f'\n All done. Plot saved under "{fn_out}"\n')
    print('\033[91m Best chunk size: {} \033[0m'.format(chunk_sizes[run_times.index(min(run_times))]))
    print(' If that looks right, please update your favorite dvas recipe accordingly !\n')

def run_recipe(rcp_fn, flights=None):
    ''' Loads and execute a dvas recipe.

    Args:
        rcp_fn (pathlib.Path): path to the specific dvas recipe to execute.
        flights (pathlib.Path, optional): path to the text file specifiying specific radiososnde
            flights to process. The file should contain one tuple of evt_id, rig_rid per line,
            e.g.::

                # This is a comment
                # Each line should contain the event_id, rig_id
                # These must be integers !
                12345, 1
                12346, 1

    '''

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
