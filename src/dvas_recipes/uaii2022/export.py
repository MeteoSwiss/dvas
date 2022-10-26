"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: export recipes for the UAII2022 campaign
"""

# Import general Python packages
import logging
import inspect
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# Import dvas modules and classes
from dvas.logger import log_func_call
from dvas.dvas import Database as DB
from dvas.environ import path_var as dvas_path_var
from dvas.version import VERSION
from dvas.data.data import MultiRSProfile, MultiGDPProfile
from dvas.hardcoded import PRF_TDT, PRF_ALT, PRF_FLG, PRF_IDX, TAG_GDP

# Import from dvas_recipes
from .. import dynamic
from ..recipe import for_each_flight
from ..errors import DvasRecipesError
from .. import utils as dru
from . import tools

logger = logging.getLogger(__name__)


@for_each_flight
@log_func_call(logger, time_it=False)
def export_profiles(tags: str | list, which: str | list):
    """ Export profiles from the db to netCDF

    Args:
        tags (str|list): list of tags to identify profiles to export in the DB.
        which (str|list): list of profiles to include, e.g.: ['mdp', 'gdp', 'cws'].

    """

    # Format the tags
    tags = dru.format_tags(tags)

    # Extract the flight info
    (eid, rid) = dynamic.CURRENT_FLIGHT
    logger.info('Exporting %s profiles for (%s;%s) ...', which, eid, rid)

    # What is the destination
    out_path = dvas_path_var.output_path
    if out_path is None:
        raise DvasRecipesError('Output path is None.')
    if not out_path.exists():
        # Be bold and create the folder, if it does not yet exist
        logger.info('Creating output path {out_path} ...')
        out_path.mkdir(parents=True)
        # Set user read/write permission
        out_path.chmod(out_path.stat().st_mode | 0o600)

    # Let's figure out which MDPs and GDPs exist for this eid/rid
    db_view = DB.extract_global_view()
    mids = db_view.loc[(db_view.eid == eid) * (db_view.rid == rid)]

    for (_, item) in mids.iterrows():

        logger.info('Processing oid: %s [%s]...', item['oid'], item['mid'])

        # Process specific mdps/gpds only if warranted
        if item['is_gdp'] and 'gdp' not in which:
            continue
        if not item['is_gdp'] and 'mdp' not in which:
            continue

        # Very well, let's create the netCDF file.
        fname = f"{dynamic.CURRENT_STEP_ID}_"
        fname += f"{item['edt'].strftime('%Y-%m-%dT%H-%M-%S')}_"
        fname += f"{item['mid'].replace('(', '').replace(')', '')}_"
        fname += f"{item['srn'].replace(' ', '')}.nc"

        rootgrp = Dataset(out_path / fname,
                          "w", format="NETCDF4")

        # Add the basic dvas info
        val = f'{datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S UTC")}- created with dvas {VERSION}'
        setattr(rootgrp, 'history', val)

        val = "/".join([item.name for item in Path(__file__).parents[:3][::-1]])
        val += f"/{Path(__file__).name}"
        val = f'{inspect.currentframe().f_code.co_name}() [{val}]'
        setattr(rootgrp, 'source', val)

        setattr(rootgrp, 'dvas.oid', f"{item['oid']}")
        setattr(rootgrp, 'dvas.srn', f"{item['srn']}")
        setattr(rootgrp, 'dvas.pid', f"{item['pid']}")
        setattr(rootgrp, 'dvas.mid', f"{item['mid']}")
        setattr(rootgrp, 'dvas.mdl_name', f"{item['mdl_name']}")
        setattr(rootgrp, 'dvas.mdl_desc', f"{item['mdl_desc']}")
        setattr(rootgrp, 'dvas.src', f"{item['src']}")
        setattr(rootgrp, 'dvas.edt', f"{item['edt']}")
        setattr(rootgrp, 'dvas.eid', f"{item['eid']}")
        setattr(rootgrp, 'dvas.rid', f"{item['rid']}")

        # TODO: time of day, start time, burst time, ... ?

        # Now get the data from the DB
        for (var_name, var) in dynamic.ALL_VARS.items():

            logger.info(f'{var_name}')

            # Assemble the search filter
            filt = tools.get_query_filter(tags_in=tags + [eid, rid],
                                          tags_out=dru.rsid_tags(pop=tags),
                                          oids=[item['oid']])

            if item['is_gdp']:
                prf = MultiGDPProfile()
                prf.load_from_db(f'and_({filt}, tags("{TAG_GDP}"))', var_name,
                                 tdt_abbr=dynamic.INDEXES[PRF_TDT],
                                 alt_abbr=dynamic.INDEXES[PRF_ALT],
                                 ucr_abbr=var['ucr'],
                                 ucs_abbr=var['ucs'],
                                 uct_abbr=var['uct'],
                                 ucu_abbr=var['ucu'],
                                 inplace=True)
            else:
                prf = MultiRSProfile()
                prf.load_from_db(f'and_({filt}, not_(tags("{TAG_GDP}")))', var_name,
                                 tdt_abbr=dynamic.INDEXES[PRF_TDT],
                                 alt_abbr=dynamic.INDEXES[PRF_ALT],
                                 inplace=True)

            if len(prf) != 1:
                raise DvasRecipesError(f'Search query returned {len(prf)} profiles instead of 1.')

            # Create the netCDF base dimensions, if it does not already exists
            if len(rootgrp.dimensions.keys()) == 0:
                rootgrp.createDimension("idx", len(prf[0].data))
                rootgrp.createDimension("time", len(prf[0].data))
                rootgrp.createDimension("ref_gph", len(prf[0].data))

                idx = rootgrp.createVariable("idx", "i8", dimensions=("idx"))
                idx[:] = prf[0].data.index.get_level_values(PRF_IDX).values
                time = rootgrp.createVariable("time", "f8", dimensions=("time"))
                time.units = 's'
                time[:] = prf[0].data.index.get_level_values(PRF_TDT).seconds.values
                ref_gph = rootgrp.createVariable("ref_gph", "f8",
                                                 dimensions=("ref_gph"))
                ref_gph.units = 'm'
                ref_gph[:] = prf[0].data.index.get_level_values(PRF_ALT).values

            # Now store all the columns
            for col in prf[0].data.columns:

                if col in [PRF_FLG]:
                    nc_tpe = 'i8'
                    np_type = 'int64'
                    na_value = 0
                else:
                    nc_tpe = 'f8'
                    np_type = 'float64'
                    na_value = np.nan
                var_nc = rootgrp.createVariable(f'{var_name}_{col}', nc_tpe,
                                                dimensions=("idx"))
                var_nc[:] = prf[0].data[col].to_numpy(dtype=np_type, na_value=na_value)

        rootgrp.close()
