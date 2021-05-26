"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level plotting for the UAII22 recipe
"""

# Import general Python packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import dvas modules and classes
from dvas.logger import recipes_logger as logger
from dvas.logger import log_func_call
from dvas.data.data import MultiRSProfile
from dvas.plots import utils as dpu

# Import from dvas_recipes
from ...errors import DvasRecipesError

@log_func_call(logger, time_it=True)
def flight_overview(eid, rid, rcp_vars, tags='sync'):
    """ Create an "overview" plot of all the recipe variables for a given flight.
    Args:
        eid (str|int): event id to be synchronized, e.g. 80611
        rid (str|int): rig id to be synchronized, e.g. 1
        rcp_vars (dict): names of variables to process, and associated uncertainties, e.g.::

            {'temp': {'ucr': 'temp_ucr', 'ucs': 'temp_ucs', 'uct': 'temp_uct', 'ucu':}}

        tags (str|list of str, optional): tag names for the search query into the database.
            Defaults to 'sync'.

    """

    # Deal with the search tags
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        raise DvasRecipesError('Ouch ! tags should be of type str|list. not: {}'.format(type(tags)))

    # What search query will let me access the data I need ?
    filt = "and_(tags('e:{}'), tags('r:{}'), {})".format(eid, rid,
                                                         "tags('" + "'), tags('".join(tags) + "')")

    plt.close()
    fig = plt.figure(figsize=(dpu.WIDTH_TWOCOL, 5))

    # Use gridspec for a fine control of the figure area.
    fig_gs = gridspec.GridSpec(len(rcp_vars), 1, height_ratios=[1]*len(rcp_vars), width_ratios=[1],
                                   left=0.08, right=0.87, bottom=0.17, top=0.9,
                                   wspace=0.05, hspace=0.05)

    for (var_ind, var_name) in enumerate(rcp_vars):


        ax = fig.add_subplot(fig_gs[var_ind, 0])

        rs_prfs = MultiRSProfile()
        rs_prfs.load_from_db(filt, var_name, 'time', alt_abbr='gph')


        for (prf_ind, prf) in enumerate(rs_prfs):

            x = getattr(prf, 'val').index.get_level_values('tdt')
            y = getattr(prf, 'val').values
            ax.plot(x/1e9, y, '-', lw=0.7, label=rs_prfs.get_info('mid')[prf_ind])


    plt.show()
