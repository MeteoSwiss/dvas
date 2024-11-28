"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This file contains the very-high-level dvas_recipe commands, to initialize and run them.
"""

# Import from Python
import logging
import multiprocessing as mpr
from datetime import datetime
import time
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs

# Import from dvas
from dvas.data.strategy.data import GDPProfile
from dvas.data.data import MultiGDPProfile
from dvas.database.database import InfoManager, DatabaseManager
from dvas.database.model import Model as TableModel
from dvas.database.model import Object as TableObject
from dvas.tools.gdps.gdps import combine
from dvas import dynamic as dyn
from dvas.environ import path_var
import dvas.plots.utils as dpu
from dvas.hardcoded import TAG_SYNC

# Import from dvas recipes
from .errors import DvasRecipesError
from .utils import default_arena_path, configs_storage_path, recipe_storage_path
from .recipe import Recipe

# Setup the local logger
logger = logging.getLogger(__name__)


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
        raise DvasRecipesError("arena_path should be of type pathlib.Path, not: " +
                               f"{type(arena_path)}")

    print(f"Initializing a new dvas processing arena under {arena_path} ...")

    # Require a new folder to avoid issues ...
    while arena_path.exists():
        arena_path = input(f'{arena_path} already exists.' +
                           ' Please specify a new (relative) path for the dvas arena:')
        arena_path = Path(arena_path)

    # Very well, setup the config files for the dvas database initialization
    shutil.copytree(configs_storage_path(), arena_path / 'configs', ignore=None,
                    dirs_exist_ok=False)

    # And also copy the dvas recipes, in case the user wants to use these
    shutil.copytree(recipe_storage_path(), arena_path, ignore=None, dirs_exist_ok=True)

    # Say goodbye ...
    print(f'All done in {(datetime.now()-start_time).total_seconds()}s.')


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
    dyn.DATA_IN_DB = True  # We also make sure to store all the data is this temporary db
    print('\n Setting-up a temporary in-memory dvas database ...')

    # Set the config file path, so that we can have a DB initialize with proper parameters.
    # Point towards the core dvas file, so that this can be run from anywhere.
    setattr(path_var, 'config_dir_path', configs_storage_path())

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
                'pid': '1'}]

    # Next prepare these custom profiles. They require oids that need to be generated by the DB
    # tables themsleves. This is were things get super manual and self-referencing ...
    gdp_prfs = []
    for (ind, item) in enumerate(db_data):
        # Get the model thingy ...
        model = db_mngr.get_or_none(
            TableModel, search={'where': TableModel.mdl_name == item['mdl_name']})

        # ... and use it to create a dedicated instrument entry in the Object Table ...
        oid = TableObject.create(srn=item['srn'], pid=item['pid'], model=model).oid
        # ... so we can add it to the db_data we assembled above ...
        db_data[ind].update({'oid': oid})

        # Prepare some datasets to play with: first an InfoManager with the oid we just minted ...
        info = InfoManager('20210616T0000Z', oid, tags={'e:1', 'r:1', TAG_SYNC})

        # And then some random data ...
        data = pd.DataFrame({'alt': np.arange(0, prf_length, 1),
                             'tdt': np.arange(0, prf_length, 1),
                             'val': np.random.rand(prf_length),
                             'flg': np.zeros(prf_length),
                             'ucs': np.random.rand(prf_length),
                             'uct': np.random.rand(prf_length),
                             'ucu': np.random.rand(prf_length)})

        # Let's assemble an actuale GDPProfile, and store it for later
        gdp_prfs += [GDPProfile(info, data)]

    # We can now build a bona fide MultiGDPProfile to test things out. Here again, we assume that
    # certain variable names have been properly defined in the db configuration files.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': 'temp', 'tdt': 'time', 'alt': 'gph', 'flg': None,
                     'ucs': 'temp_ucs', 'uct': 'temp_uct', 'ucu': 'temp_ucu'},
                    gdp_prfs)

    print('\n Computing the weighted-mean of 3 profiles with different chunk sizes,' +
          f' with {n_cpus} cpus ...\n')

    # We are now ready to launch the combination routine, which relies on Profile slicing.
    run_times = []
    # What chunk sizes need to be tested ?
    chunk_sizes = np.linspace(chunk_min, chunk_max, n_chunk, dtype=int)
    chunk_sizes = list(chunk_sizes) + list(chunk_sizes)[::-1]
    # Loop through all them them twice in reverse orders (just to get two data points for each).
    for chunk_size in chunk_sizes:
        start_time = time.perf_counter()
        # Here I don't actually care about the outcome, just the time it takes to get there ...
        _ = combine(multiprf, binning=1, method='weighted arithmetic mean', chunk_size=chunk_size,
                    n_cpus=n_cpus)
        # Store this info, and also print it to screen ...
        run_times += [time.perf_counter() - start_time]
        print(f'  chunk_size: {chunk_size} <=> {run_times[-1]:.2f}s')

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

    fn_out = f'dvas_optimize_{datetime.now().strftime("%Y%m%dT%H%M%S")}_{n_cpus}-cpus.pdf'

    plt.savefig(fn_out)

    print(f'\n All done. Plot saved under "{fn_out}"\n')
    print(f'\033[91m Best chunk size: {chunk_sizes[run_times.index(min(run_times))]} \033[0m')
    print(' If that looks right, please update your favorite dvas recipe accordingly !\n')


def run_recipe(rcp_fn: Path, fid_log_fn: Path,
               fid_to_treat: list = None, from_step_id: str = None, until_step_id: str = None,
               debug: bool = False):
    ''' Loads and execute a dvas recipe.

    Args:
        rcp_fn (pathlib.Path): path to the specific dvas recipe to execute.
        fid_log_fn (pathlib.Path): path to the log linking flight ids, event ids, rig ids,
            and event datetimes.
        fid_to_treat (list of str, optional): list of flights ids to treat specifically.
        from_step_id (str, optional): if set, will skip all processing until this step_id value.
            Defaults to None.
        until_step_id (str, optional): if set, will skip all processing step beyond this step_id
            value. Defaults to None.
        debug (bool, optional): if True, will force-set the logging level to DEBUG.
            Defaults to False.


        The "fid_log_fn" file should contain one tuple of flight if, event id, rig id, and
        event datetime per line, e.g.::

                # This is a comment
                F01,e:12345,r:1,2022-08-19T19:00:00.000Z
                T04,e:12340,r:2,2022-08-19T15:45:00.000Z

        Event ids are meant to be GRUAN flight ids, in order to link GDPs with specific flights.

        Rig ids are also extracted from GRUAN parameters in GDP files to distinguish different rigs
        in case of multiple simultaneous launches.

        Event datetimes are not used by dvas directly, but allow for an easier identification of
        flights in UPPS (UAII Plot Preview Software).

        Flight ids are user-defined reference strings, used to identify specific flight in a given
        campaign - these are typically defined before GRUAN event ids (that are generated at the
        creation of GDPs).)

        TODO:

            At present, it is not possible to link fids to eids, because the information is missing
            from the iMS100 GDP. This products would require the field  'g.measurement.InternalKey'
            to be present, like the RS41 GDP, e.g.:

            :g.Measurement.InternalKey = "UAII2022_F08"

    '''

    # Make sure the user specified info is valid.
    for this_path in [rcp_fn, fid_log_fn]:
        if not isinstance(this_path, Path):
            raise DvasRecipesError(f'Bad pathlib.Path format: {this_path}')
        if not this_path.exists():
            raise DvasRecipesError(f'Inexistant path: {this_path}')

    # Extract the flight log
    fid_eid_log = np.atleast_2d(np.genfromtxt(fid_log_fn, comments='#', delimiter=',',
                                              dtype=str, autostrip=True))

    # If warranted, let's sub-select the flights to process from the list provided
    # This is all a bit convoluted, but meant to ease the processing of subsequebnt flights
    # individually in a campaign setting
    if fid_to_treat is not None:
        fid_eid_log = [item for item in fid_eid_log if item[0] in fid_to_treat]

    # Extract the eids - rids tuples required by dvas, and drop the rest
    eids_to_treat = [(item[0], f'e:{item[1]}', f'r:{item[2]}') for item in fid_eid_log]
    if len(eids_to_treat) == 0:
        eids_to_treat = None  # With this, we shall treat whatever is present in the db

    # Very well, I am now ready to start initializing the recipe.
    rcp = Recipe(rcp_fn, eids_to_treat=eids_to_treat, debug=debug)

    starttime = datetime.now()

    # Launch the procesing !
    rcp.execute(from_step_id=from_step_id, until_step_id=until_step_id)

    # All done !
    logger.info('\n\n All done - "%s" recipe completed in %is.\n',
                rcp.name, (datetime.now()-starttime).total_seconds())
