"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This file contains high-level (common) recipe utilities.

"""

from ruamel.yaml import YAML

# Load dvas-specific things
from dvas.environ import path_var
from dvas.dvas import Log
import dvas.plots.utils as dpu

# Setup the YAML loader
yaml = YAML(typ='safe')

def initialize_recipe(rcp_fpath):
    """ High-level function that initializes high-level recipe parameters set in a dedicated YML
    file.

    It is assumed that the recipe parameters are stored in a .yml file with the same name as the
    recipe. I.e. for the `multi_gdps_recipe.py` file, the parameters **must** be stored in
    `multi_gdps_recipe.yml`.

    Args:
        rcp_fpath (pathlib.Path): path+name of the recipe file.

    """

    # Load the recipe parameters
    rcp_params = yaml.load(rcp_fpath.with_suffix('.yml'))

    # Assign the paths
    for path in ['config_dir_path', 'orig_data_path', 'local_db_path', 'output_path']:
        val = rcp_params['paths'][path.split('_path')[0]]
        if val[0] == '/':
            setattr(path_var, path, val)
        else:
            setattr(path_var, path, rcp_fpath.parent / val)

    # Start the logging
    Log.start_log(rcp_params['general']['log_mode'], level=rcp_params['general']['log_lvl'])

    # Let us fine-tune the plotting behavior of dvas if warranted
    if rcp_params['general']['do_latex']:
        dpu.set_mplstyle('nolatex') # The safe option. Use 'latex' fo prettier plots.

    # The generic formats to save the plots in
    dpu.PLOT_FMTS = rcp_params['general']['plot_fmts']

    # Show the plots on-screen ?
    dpu.PLOT_SHOW = rcp_params['general']['plot_show']
