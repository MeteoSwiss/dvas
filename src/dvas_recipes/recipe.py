"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This file contains specific recipe classes and utilities.

"""

# Import generic packages
from importlib import import_module
from multiprocessing import cpu_count
from pathlib import Path
from ruamel.yaml import YAML
import numpy as np

# Import from dvas
from dvas.environ import path_var
from dvas.logger import recipes_logger as logger
from dvas.dvas import Log
from dvas.dvas import Database as DB
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME
import dvas.plots.utils as dpu
from dvas import dynamic as dyn

# Import from dvas_recipesz
from .errors import DvasRecipesError
from . import dynamic as rcp_dyn

# Setup the YAML loader
yaml = YAML(typ='safe')

def for_each_flight(func):
    """ This function is meant to be used as a decorator around recipe step functions that need to
    be executed sequentially for all the specified flights.

    Args:
        func (function): the recipe step function to decorate.

    """

    def wrapper(**kwargs):
        """ A small wrapper function that loops over dynamics.ALL_FLIGHTS and updates the value of
        dynamic.THIS_FLIGHT before calling the wrapped function (that is assumed will itself call
        dynamic.THIS_FLIGHT directly).

        Note:
            This decorator can be used in conjunction with `for_each_var`.

        Args:
            **kwargs: keyword arguments of the decorated function, that ought to be defined in the
                `.rcp` YAML file by the end user.
        """

        # Loop through each flight, and apply func as intended, after updating the value of
        # dynamic.THIS_FLIGHT
        for flight in rcp_dyn.ALL_FLIGHTS:
            rcp_dyn.CURRENT_FLIGHT = flight
            func(**kwargs)

    return wrapper

def for_each_var(func):
    """ This function is meant to be used as a decorator around recipe step functions that need to
    be executed sequentially for all the specified variables.

    Args:
        func (function): the recipe step function to decorate.

    """

    def wrapper(**kwargs):
        """ A small wrapper function that loops over dynamic.ALL_VARS and updates the value of
        dynamic.THIS_VAR before calling the wrapped function (that is assumed will itself call
        dynamic.THIS_VAR directly).

        Note:
            This decorator can be used in conjunction with `for_each_flight`.

        Args:
            **kwargs: keyword arguments of the decorated function, that ought to be defined in the
                `.rcp` YAML file by the end user.
        """

        # Loop through each flight, and apply func as intended, after updating the value of
        # dynamic.THIS_FLIGHT
        for  var in rcp_dyn.ALL_VARS:
            rcp_dyn.CURRENT_VAR = var
            func(**kwargs)

    return wrapper


class RecipeStep:
    """ RecipeStep class, to handle individual recipe steps and their execution """

    _step_id = None # str: to identify the recipe step
    _name = None # str: to identify even more easily the recipe step
    _run = None # bool: whether to run the step, or not
    _func = None # actual function associated to the recipe. This is what will do the work
    _kwargs = None # dict: keywords arguments associated to the recipe

    def __init__(self, func, step_id, name, run, kwargs):
        """ RecipeStep initialization function.

        Args:
            func (function): dvas_recipe function to be launched as part of the step.
            step_id (str|int): id of the recipe step, e.g. '01a'.
            name (str): name of the recipe step, e.g. 'uaii2022.sync.sync_flight'
            run (bool): if True, the step is supposed to be executed as part of the recipe.
            kwargs (dict): dictionnary of keyword arguments to be fed to func.
        """

        self._func = func
        self._step_id = step_id
        self._name = name
        self._run = run
        self._kwargs = kwargs

    @property
    def step_id(self):
        """ I.D. of this recipe step """
        return self._step_id

    @property
    def name(self):
        """ name of this recipe step """
        return self._name

    @property
    def run(self):
        """ Whether this step should be executed, or not, as part of the recipe """
        return self._run

    def execute(self):
        """ Launches the processing associated with this recipe step.
        """

        # Store the step_id in the dynamic module for easy access by the function if needed
        rcp_dyn.CURRENT_STEP_ID = self.step_id

        # Actually launch the function, which may be decorated (but I don't need to know that !)
        logger.info('Executing recipe step %s: %s', self.step_id, self.name)
        self._func(**self._kwargs)

class Recipe:
    """ Recipe class, designed to handle the loading/initialization/execution of dvas recipes from
    dedicated YAML files.
    """

    _name = None
    _steps = None

    def __init__(self, rcp_fn, flights=None):
        """ Recipe initialization from a suitable YAML recipe file.

        Args:
            rcp_fn (pathlib.Path): path of the recipe file to initialize.
            flights (2D ndarray of int, optional): ndarray listing specific flights to be processed,
                using their event id and rig id for identification.
                Defaults to None = all available. Array must be 2D, e.g.::

                    [[12345, 1], [12346, 1]]

        """

        # Load the recipe data
        rcp_data = yaml.load(rcp_fn)

        # Set the recipe name
        self._name = rcp_data['rcp_name']

        # Setup the dvas paths
        for item in rcp_data['rcp_paths'].items():
            if item[1][0] == '/':
                setattr(path_var, item[0], item[1])
            else:
                setattr(path_var, item[0], rcp_fn.parent / item[1])

        # Of all the paths, it is essential to make sure that orig_data_path exists.
        # Else, en empty database will be created (without complaints), and it will be hard for the
        # user to understand that it is empty because no data was found (because the path was off).
        # Fixes #165.
        if not Path(path_var.orig_data_path).exists():
            raise DvasRecipesError(f'Ouch ! the following orig_data_path (defined in {rcp_fn}) '+
                                   'does not exist:' +
                                   f' {path_var.orig_data_path}')

        # Start the dvas logging
        Log.start_log(rcp_data['rcp_params']['general']['log_mode'],
                      level=rcp_data['rcp_params']['general']['log_lvl'])

        # Let us fine-tune the plotting behavior of dvas if warranted
        if rcp_data['rcp_params']['general']['do_latex']:
            dpu.set_mplstyle('latex')
        else:
            dpu.set_mplstyle('nolatex')

        # The generic formats to save the plots in
        dpu.PLOT_FMTS = rcp_data['rcp_params']['general']['plot_fmts']
        # Show the plots on-screen ?
        dpu.PLOT_SHOW = rcp_data['rcp_params']['general']['plot_show']

        # Adjust the dvas chunk size
        rcp_dyn.CHUNK_SIZE = rcp_data['rcp_params']['general']['chunk_size']

        # Set the number of cpus.
        rcp_dyn.N_CPUS = rcp_data['rcp_params']['general']['n_cpus']
        if rcp_dyn.N_CPUS is None or rcp_dyn.N_CPUS > cpu_count():
            rcp_dyn.N_CPUS = cpu_count()

        # Store the index names
        rcp_dyn.INDEXES = rcp_data['rcp_params']['index']

        # Store the variables to be processed and their associated uncertainties.
        rcp_dyn.ALL_VARS = rcp_data['rcp_params']['vars']

        # Get started with the initializations of the different recipe steps
        self._steps = []
        for item in rcp_data['rcp_steps'].items():

            # Danger zone: here I access the correct function by importing the corresponding
            # recipe module (from scratch ?)... this seems to work ... until proven otherwise ?
            rcp_mod = import_module('.'+'.'.join(item[0].split('.')[:-1]), 'dvas_recipes')

            # Initialize the recipe step, and add it to the list
            self._steps += [RecipeStep(getattr(rcp_mod, item[0].split('.')[-1]),
                                       item[1]['step_id'], item[0],
                                       item[1]['run'], item[1]['kwargs'])]

        # Set the flights to be processed, if warranted.
        # These get stored in the dedicated "dynamic" module, for easy access everywhere.
        if flights is not None:
            if ndim:=np.ndim(flights) != 2:
                raise DvasRecipesError('Ouch ! np.ndim(flights) should be 2, not: {}'.format(ndim))

            rcp_dyn.ALL_FLIGHTS = flights

    @property
    def name(self):
        """ Returns the name of the Recipe. """
        return self._name

    @property
    def n_steps(self):
        """ Returns the number of steps inside the Recipe. """
        return len(self._steps)

    def init_db(self):
        """ Initialize the dvas database, and fetch the raw data required for the recipe. """

        # Here, make sure the DB is stored locally, and not in memory.
        dyn.DB_IN_MEMORY = False

        # Use this command to clear the DB
        DB.refresh_db()

        # Init the DB
        DB.init()

        # Fetch the raw data
        DB.fetch_raw_data([rcp_dyn.INDEXES[PRF_REF_TDT_NAME]] +
                          [rcp_dyn.INDEXES[PRF_REF_ALT_NAME]] +
                          list(rcp_dyn.ALL_VARS) +
                          [rcp_dyn.ALL_VARS[var][uc] for var in rcp_dyn.ALL_VARS
                           for uc in rcp_dyn.ALL_VARS[var]],
                          strict=True)

    def get_all_flights_from_db(self):
        """ Identifies all the radiosonde flights present in the dvas database.

        Returns:
            2D ndarray of int: the array of event_id and rig_id for each flight, with the following
                structrure::

                    [[12345, 1], [12346, 1]]
        """

        raise Exception('Fix ME !')

        return 0

    def execute(self, from_step_id=None):
        """ Run the recipe step-by-step, possibly skipping some of the first ones.

        Args:
            from_step_id (str|int, optional): if set, will start the processing from this specific
                step_id. Defaults to None = start at first step.

        """

        # First, we setup the dvas database
        self.init_db()

        # If warranted, find all the flights that need to be processed.
        if rcp_dyn.ALL_FLIGHTS is None:
            rcp_dyn.ALL_FLIGHTS = self.get_all_flights_from_db()

        # Now that everything is in place, all that is required at this point is to launch each step
        # one after the other. If warranted, lock the execution of steps until a certain one is
        # reached.
        unlock_steps = from_step_id is None

        # Loop through the step list ...
        for step in self._steps:

            # If we reach the starting step, unlock the processing
            if not unlock_steps and step.step_id == from_step_id:
                unlock_steps = True

            # If warranted, execute the step
            if unlock_steps and step.run:
                step.execute()
