"""
Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This file contains specific recipe classes and utilities.

"""

# Import generic packages
import logging
from importlib import import_module
from multiprocessing import cpu_count
from pathlib import Path
from ruamel.yaml import YAML
import numpy as np

# Import from dvas
from dvas.environ import path_var
from dvas import __version__ as dvas_version
from dvas.dvas import Log
from dvas.dvas import Database as DB
from dvas.hardcoded import PRF_TDT, PRF_ALT, FLG_PRM_NAME_SUFFIX
import dvas.plots.utils as dpu
from dvas import dynamic as dyn

# Import from dvas_recipes
from .errors import DvasRecipesError
from . import dynamic as rcp_dyn

# Setup the local logger
logger = logging.getLogger(__name__)

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
            logger.info('Processing flight: %s', flight)
            func(**kwargs)

    return wrapper


def for_each_var(incl_wdir=True, incl_wspeed=True, incl_wvec=False, incl_latlon=False):
    """ Parameteric decorator, to enable the looping over different variables, with the option
    to select different wind elements.

    Args:
        incl_wdir (bool): whether to include the wdir variable, or not.
        incl_wspeed (bool): whether to include the wspeed variable, or not.
        incl_wvec (bool): whether to include the wvec variable, or not.
        incl_latlon (bool): whether to include the lat and lon variables, or not.


    """

    def for_each_selected_var(func):
        """ This function is the actual decorator to be used around recipe step functions that need
        to be executed sequentially for all the CWS variables.

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
                **kwargs: keyword arguments of the decorated function, that ought to be defined in
                    the `.rcp` YAML file by the end user.
            """

            # Loop through each flight, and apply func as intended, after updating the value of
            # dynamic.THIS_FLIGHT
            for var in rcp_dyn.ALL_VARS:
                # Apply the parameteric choices
                if var == 'wvec' and not incl_wvec:
                    continue
                if var == 'wdir' and not incl_wdir:
                    continue
                if var == 'wspeed' and not incl_wspeed:
                    continue
                if var in ['lat', 'lon'] and not incl_latlon:
                    continue
                rcp_dyn.CURRENT_VAR = var
                logger.info('Processing variable: %s', var)
                func(**kwargs)

        return wrapper
    return for_each_selected_var


def for_each_oscar_var(func):
    """ This function is meant to be used as a decorator around recipe step functions that need to
    be executed sequentially for all the OSCAR variables.

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
        for var in rcp_dyn.ALL_VARS:
            if var in ['wdir', 'speed']:
                continue
            rcp_dyn.CURRENT_VAR = var
            logger.info('Processing variable: %s', var)
            func(**kwargs)

    return wrapper


class RecipeStep:
    """ RecipeStep class, to handle individual recipe steps and their execution """

    _step_id = None  # str: to identify the recipe step
    _name = None  # str: to identify even more easily the recipe step
    _run = None  # bool: whether to run the step, or not
    _func = None  # actual function associated to the recipe. This is what will do the work
    _kwargs = None  # dict: keywords arguments associated to the recipe

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

        # Deal with the cases where no kwargs were speficied
        if kwargs is None:
            kwargs = {}

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
        logger.info('$SFLASHSTART RECIPE STEP %s: %s $EFLASH',
                    self.step_id, self.name)
        self._func(**self._kwargs)


class Recipe:
    """ Recipe class, designed to handle the loading/initialization/execution of dvas recipes from
    dedicated YAML files.
    """

    _name = None
    _steps = None
    _reset_db = True

    def __init__(self, rcp_fn, eids_to_treat=None, debug=False):
        """ Recipe initialization from a suitable YAML recipe file.

        Args:
            rcp_fn (pathlib.Path): path of the recipe file to initialize.
            eids_to_treat (list, optional): list of ('fid', 'e:eid', 'r:rid') tuples.
                If None, will process all the flights found in the DB.
            debug (bool, optional): if True, will force-set the logging level to DEBUG.
                Defaults to False.

        """

        # Load the recipe data
        rcp_data = yaml.load(rcp_fn)

        # Set the recipe name
        self._name = rcp_data['rcp_name']
        rcp_dyn.RECIPE = rcp_data['rcp_name']
        # Set whether we want to reset the DB, or load an existing one
        self._reset_db = rcp_data['rcp_params']['general']['reset_db']
        # Set whether the Profile Data is stored in the DB, or to disk
        self._data_in_db = rcp_data['rcp_params']['general']['data_in_db']

        # Setup the dvas paths
        for item in rcp_data['rcp_paths'].items():
            # Path anchors are just used for avoid duplicating stuff in the YAML file
            if item[0] == 'path_anchors':
                continue
            # Set the path
            setattr(path_var, item[0], Path(item[1]['base_path']) / item[1]['sub_path'])

        # Set the flights to be processed, if warranted.
        # These get stored in the dedicated "dynamic" module, for easy access everywhere.
        if eids_to_treat is not None:

            # Get the list of fids, in order to create dedicated folders
            fids = '_'.join([item[0] for item in eids_to_treat])

            # Adjust the input and output paths accordingly
            setattr(path_var, 'output_path', path_var.output_path / fids)
            setattr(path_var, 'orig_data_path', path_var.orig_data_path)

            rcp_dyn.ALL_FLIGHTS = eids_to_treat

        # Of all the paths, it is essential to make sure that orig_data_path exists.
        # Else, an empty database will be created (without complaints), and it will be hard for the
        # user to understand that it is empty because no data was found (because the path was off).
        # Fixes #165.
        if not Path(path_var.orig_data_path).exists():
            raise DvasRecipesError(f'orig_data_path not exist: {path_var.orig_data_path}')

        # Start the dvas logging
        # Any path business should be completed by this point, as the log will check and create
        # some of them, including the output path.
        if debug:
            loglvl = 'DEBUG'
        else:
            loglvl = rcp_data['rcp_params']['general']['log_lvl']
        Log.start_log(rcp_data['rcp_params']['general']['log_mode'],
                      level=loglvl)

        logger.info('Launching the %s recipe with dvas v%s', self._name, dvas_version)
        logger.info('orig_data_path was set to: %s', path_var.orig_data_path)
        logger.info('output_path was set to: %s', path_var.output_path)

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
        for item in rcp_data['rcp_steps']:

            # Danger zone: here I access the correct function by importing the corresponding
            # recipe module (from scratch ?)... this seems to work ... until proven otherwise ?
            rcp_mod = import_module('.'+'.'.join(item['fct'].split('.')[:-1]), 'dvas_recipes')

            # Initialize the recipe step, and add it to the list
            self._steps += [RecipeStep(getattr(rcp_mod, item['fct'].split('.')[-1]),
                                       item['step_id'], item['fct'],
                                       item['run'], item['kwargs'])]

        # Keep a list of all the steps ids. WIll be useful for auto-tagging
        rcp_dyn.ALL_STEP_IDS = [item._step_id for item in self._steps]

    @property
    def name(self):
        """ Returns the name of the Recipe. """
        return self._name

    @property
    def n_steps(self):
        """ Returns the number of steps inside the Recipe. """
        return len(self._steps)

    @staticmethod
    def init_db(reset: bool = True, data_in_db: bool = True):
        """ Initialize the dvas database, and fetch the original data required for the recipe.

        Args:
            reset (bool, optional): if True, the DB will be filled from scratch. Else, only new
                original data will be ingested. Defaults to True.
        """

        # Here, make sure the DB is stored locally, and not in memory.
        dyn.DB_IN_MEMORY = False
        # Set whether the Profile data is stored in the DB, or on external text files
        dyn.DATA_IN_DB = data_in_db

        if reset:
            logger.info("Resetting the DB ...")
            # Use this command to clear the DB
            DB.refresh_db()

        # Init the DB
        DB.init()

        # Fetch the original data
        DB.fetch_original_data([rcp_dyn.INDEXES[PRF_TDT]] +  # The reference times
                               [rcp_dyn.INDEXES[PRF_ALT]] +  # The reference altitudes
                               list(rcp_dyn.ALL_VARS) +  # All the primary variables
                               # All the flags associated to the primary variables
                               [f'{item}{FLG_PRM_NAME_SUFFIX}' for item in rcp_dyn.ALL_VARS] +
                               # All the necessary uncertainties
                               [rcp_dyn.ALL_VARS[var][uc] for var in rcp_dyn.ALL_VARS
                                for uc in rcp_dyn.ALL_VARS[var]], strict=True)

    @staticmethod
    def get_all_flights_from_db():
        """ Identifies all the radiosonde flights present in the dvas database.

        Returns:
            2D ndarray of str: the array of event_id and rig_id tags for each flight, with the
                following structrure::

                    [['e:12345', 'r:1'], ['e:12346', 'r:1']]
        """

        # Load the DB, and get the list of all the oids in there
        global_view = DB.extract_global_view()

        flights = np.array([[item[1]['eid'], item[1]['rid']] for item in global_view.iterrows()])

        # Drop the duplicates
        flights = np.unique(flights, axis=0)

        # Make some quick test to make sure I did not mess anyting up
        assert np.ndim(flights) == 2
        assert np.shape(flights)[1] == 2

        logger.info('Found %i flight(s) in the database.', len(flights))

        return flights

    def execute(self, from_step_id=None, until_step_id=None):
        """ Run the recipe step-by-step, possibly skipping some of the first ones.

        Args:
            from_step_id (str|int, optional): if set, will start the processing from this specific
                step_id. Defaults to None = start at first step.
            until_step_id (str|int, optional): if set, will end the processing after this specific
                step_id. Defaults to None = go until the end of the recipe.

        """

        # If I am skipping any steps, let's disable the DB reset. Else, it will blow up in my face.
        if from_step_id is not None:
            logger.info('Force-disable the DB reset, and skip until the recipe step: %s',
                        from_step_id)
            self._reset_db = False

        # First, we setup the dvas database
        self.init_db(reset=self._reset_db, data_in_db=self._data_in_db)

        # If warranted, find all the flights that need to be processed.
        if rcp_dyn.ALL_FLIGHTS is None:
            rcp_dyn.ALL_FLIGHTS = self.get_all_flights_from_db()
        else:
            logger.info('Processing %i flight(s).', len(rcp_dyn.ALL_FLIGHTS))

        # Raise an error if no flights were specified/foudn in the DB
        if len(rcp_dyn.ALL_FLIGHTS) == 0:
            raise DvasRecipesError('No flights to process !')

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

            # If we reached the final step, lock all the subsequent ones
            if step.step_id == until_step_id:
                unlock_steps = False
