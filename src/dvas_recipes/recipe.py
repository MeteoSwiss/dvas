"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This file contains specific recipe classes and utilities.

"""

# Import generic packages
from importlib import import_module
from ruamel.yaml import YAML

# Import from dvas
from dvas.environ import path_var
from dvas.dvas import Log
from dvas.dvas import Database as DB
import dvas.plots.utils as dpu

# Import from dvas_recipes
from .hardcoded import DVAS_RECIPE_NAMES
from .errors import DvasRecipesError

# Setup the YAML loader
yaml = YAML(typ='safe')

class RecipeStep:
    """ RecipeStep class, to handle individual recipe steps and their execution """

    _step_id = None
    _run = None
    _func = None
    _args = None

    def __init__(self, func, step_id, run, args):
        """ RecipeStep initialization function.

        Args:
            func (function): dvas_recipe function to be launched as part of the step.
            step_id (str|int): id of the recipe step, e.g. '01a'.
            run (bool): if True, the step is supposed to be executed as part of the recipe.
            args (dict): dictionnary of arguments to be fed to func.
        """

        self._func = func
        self._step_id = step_id
        self._run = run
        self._args = args

    @property
    def step_id(self):
        """ I.D. of this recipe step """
        return self._step_id

    @property
    def run(self):
        """ Whether this step should be executed, or not, as part of the recipe """
        return self._run

    def execute(self):
        """ Launches the processing associated with this recipe step. """
        #return self._func(**self._args)
        import pdb
        pdb.set_trace()
        print(self.step_id, self.run)

class Recipe:
    """ Recipe class, designed to handle the loading/initialization/execution of dvas recipes from
    dedicated YAML files.
    """

    _name = None
    _steps = None
    _vars = None
    _index = None

    def __init__(self, rcp_fn):
        """ Recipe initialization from a suitable YAML recipe file.

        Args:
            rcp_fn (pathlib.Path): path of the recipe file to initialize.

        """

        # Load the recipe data
        rcp_data = yaml.load(rcp_fn)

        # Set the recipe name
        self._name = rcp_data['rcp_name']
        if self._name not in DVAS_RECIPE_NAMES:
            raise DvasRecipesError('Ouch ! Recipe name must absolutely be one of: ' +
                                    '{}'.format(DVAS_RECIPE_NAMES))

        # Setup the dvas paths
        for item in rcp_data['rcp_paths'].items():
            if item[1][0] == '/':
                setattr(path_var, item[0], item[1])
            else:
                setattr(path_var, item[0], rcp_fn.parent / item[1])

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

        # Store the index names
        self._index = [rcp_data['rcp_params']['index']['tdt'] +
                       rcp_data['rcp_params']['index']['alt']]

        # Store the variables to be processed and their associated uncertainties.
        self._vars = rcp_data['rcp_params']['vars']

        # Get started with the initializations of the different recipe steps
        rcp_steps = []
        for item in rcp_data['rcp_steps'].items():

            # Danger zone: here I access the correct function by importing the corresponding
            # recipe module (from scratch ?)... this seems to work ... until proven otherwise ?
            rcp_mod = import_module('.'+'.'.join(item[0].split('.')[:-1]), 'dvas_recipes')

            # Initialize the recipe step
            rcp_steps += [RecipeStep(getattr(rcp_mod, item[0].split('.')[-1]),
                                    item[1]['step_id'], item[1]['run'], item[1]['args'])]

        self._steps =  rcp_steps

        import pdb
        pdb.set_trace()

    @property
    def name(self):
        """ Returns the name of the Recipe. """
        return self._name

    @property
    def n_steps(self):
        """ Returns the number of steps inside the Recipe. """
        return len(self._steps)

    @property
    def n_vars(self):
        """ Returns the numbers of variables included in the Recipes. """
        return len(self._vars)

    def init_db(self):
        """ Initialize the dvas database, and fetch the raw data required for the recipe. """

        # Use this command to clear the DB
        DB.clear_db()

        # Init the DB
        DB.init()

        # Fetch the raw data
        DB.fetch_raw_data(list(self._index) +
                          list(self._vars) +
                          [self._vars[var][uc] for var in self._vars for uc in self._vars[var]],
                          strict=True)

    def execute(self, from_step=0):
        """ Run the recipe step-by-step, possibly skipping some of the first ones. """

        # First, we setup the dvas database
        self.init_db()

        # Now that everything is in place, all that is required at this point is to launch each step
        # one after the other.
        for step in self._steps:
            step.execute()
