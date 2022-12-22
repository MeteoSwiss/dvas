"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level OSCAR-related routines for the UAII 2022 recipe
"""

# Import from python
import logging
import functools
import multiprocessing as mp
import numpy as np
import pandas as pd
from netCDF4 import Dataset # noqa pylint: disable=E0611

# Import from dvas
from dvas.logger import log_func_call
from dvas.environ import path_var as dvas_path_var
from dvas.data.data import MultiDeltaProfile
from dvas.dvas import Database as DB
from dvas.tools.chunks import process_chunk
from dvas.hardcoded import PRF_ALT, PRF_VAL, PRF_UCR, PRF_UCS, PRF_UCT, PRF_UCU, PRF_FLG
from dvas.hardcoded import TAG_DTA, TAG_CWS

# Import from dvas_recipes
from ..errors import DvasRecipesError
from .. import dynamic
from . import tools
from .. import utils as dru
from .export import add_cf_attributes, set_attribute

# Setup local logger
logger = logging.getLogger(__name__)


@log_func_call(logger)
def compute_oscar(start_with_tags, mids=None, suffix='', institution='',
                  gph_min=0, gph_bin_size=100, gph_bin_count=350):
    """ Highest-level recipe function responsible for assembling OSCAR profiles, and storing the
    result in a dedicated netCDF.

    Args:
        start_with_tags (str|list of str): tag name(s) for the search query into the database.
        mids (list, optional): list of 'mid' to process. Defaults to None = all.
        suffix (str, optional): name of the netCDF file suffix. Defaults to ''.
        institution (str, optional): for the netCDF eponym field. Defaults to ''.
        gph_min (list, optional): min gph altitude to consider, in m. Defaults to 0.
        gph_bin_size (int, optional): gph bin size, in m. Defaults to 100.
        gph_bin_count (int, optional): gph bin count. Defaults to 350.

    """

    # Cleanup the tags
    prf_tags = dru.format_tags(start_with_tags)

    # Very well, let us first extract the 'mid', if they have not been provided
    db_view = DB.extract_global_view()
    if mids is None:
        mids = db_view.mid.unique().tolist()

    # Basic sanity check of mid
    if not isinstance(mids, list):
        raise DvasRecipesError(f'Ouch ! I need a list of mids, not: {mids}')

    # Very well, let's now loop through these, and compute the OSCAR profiles
    for mid in mids:

        # Second sanity check - make sure the mid is in the DB
        if mid not in db_view.mid.unique().tolist():
            raise DvasRecipesError(f'mid unknown: {mid}')
        else:
            logger.info('Processing %s ...', mid)

        # Create the netCDF that will be used to store the data
        fname = '_'.join([suffix, 'big-lambda', mid]) + '.nc'

        # What is the destination for the nc files ?
        out_path = dvas_path_var.output_path
        if out_path is None:
            raise DvasRecipesError('Output path is None.')
        if not out_path.exists():
            raise DvasRecipesError('Output path does not exist.')

        # Setup the root group
        rootgrp = Dataset(out_path / fname, "w", format="NETCDF4")
        # Add CF-1.7 Global Attributes
        add_cf_attributes(rootgrp, title='Big Lambda atmospheric profile',
                          institution=institution,
                          comment='See the UAII 2022 Final Report for details.')

        # Now let's assemble the list of gph bins for the high-resolution Lambda profile
        gph_bins = np.linspace(gph_min, gph_min+gph_bin_size*gph_bin_count, gph_bin_count+1)
        gph_pts = np.diff(gph_bins)/2. + gph_bins[:-1]

        # Add this right away as a dimension to the netCDF
        rootgrp.createDimension("ref_alt", len(gph_pts))
        gph_dim = rootgrp.createVariable("ref_alt", "f8", dimensions=("ref_alt"))
        gph_dim[:] = gph_pts
        gph_dim.units = 'm'
        gph_dim.standard_name = 'reference_altitude'
        gph_dim.long_name = 'Reference altitude'
        gph_dim.comment = 'Mean geopotential altitude of the successive height bins.'

        set_attribute(rootgrp, 'd.biglambda.gph_min', f'{gph_min} m')
        set_attribute(rootgrp, 'd.biglambda.gph_bin_size', f'{gph_bin_size} m')

        # What search query will let me access the data I need ?
        prf_filt = tools.get_query_filter(tags_in=prf_tags + [TAG_DTA],
                                          tags_out=dru.rsid_tags(pop=prf_tags) + [TAG_CWS],
                                          mids=[mid])

        # Start looking for all the variables
        for (var_name, var) in dynamic.ALL_VARS.items():

            # Load the delta profiles as Profiles (and not RSProfiles) since we're about to drop the
            prfs = MultiDeltaProfile()
            prfs.load_from_db(prf_filt, var_name,
                              alt_abbr=dynamic.INDEXES[PRF_ALT],
                              ucr_abbr=var['ucr'],
                              ucs_abbr=var['ucs'],
                              uct_abbr=var['uct'],
                              ucu_abbr=var['ucu'],
                              inplace=True)
            logger.info('Found %i delta_profiles for %s', len(prfs), var_name)

            # Let's extract the data I care about
            pdf = prfs.get_prms([PRF_ALT, PRF_VAL, PRF_FLG, PRF_UCR, PRF_UCS, PRF_UCT, PRF_UCU,
                                'uc_tot'], mask_flgs=None, with_metadata=['oid', 'mid', 'eid', 'rid'],
                                pooled=True)

            # Sort by altitudes
            pdf.sort_values((0, PRF_ALT), inplace=True)

            # Ready to build some chunks
            chunks = [pdf.loc[(pdf[(0, PRF_ALT)] >= item) * (pdf[(0, PRF_ALT)] < gph_bins[ind+1])]
                      for (ind, item) in enumerate(gph_bins[:-1])]

            # Start processing chunks
            proc_func = functools.partial(process_chunk, method='biglambda')
            if dynamic.N_CPUS == 1:
                proc_chunks = map(proc_func, chunks)
            else:

                # TODO: this is a bug that I do not understand. When running the script from an
                # ipython session multiple times in a row, the __spec__ variable is only defined
                # the first time around. Having this set to anything (=None when run interactively)
                # is crucial to the multiprocessing Pool routine.
                # So for now, use a terribly dangerous workaround that I do not understand.
                # This should definitely be fixed. Or not ?
                # See #121
                import sys
                try:
                    sys.modules['__main__'].__spec__ is None
                except AttributeError:
                    logger.warning('BUG: __spec__ is not set. Fixing it by hand ...')
                    sys.modules['__main__'].__spec__ = None

                pool = mp.Pool(processes=dynamic.N_CPUS)
                proc_chunks = pool.map(proc_func, chunks)
                pool.close()
                pool.join()

            # Extract the profile chunks, in order to stich them up together.
            biglambda_ms = [item[0] for item in proc_chunks]
            biglambda_ms = pd.concat(biglambda_ms, axis=0)

            # Store all this information in the netCDF
            val_nc = rootgrp.createVariable(f'{var_name}', 'f8', dimensions=("ref_alt"))
            uc_nc = rootgrp.createVariable(f'{var_name}_uc', 'f8', dimensions=("ref_alt"))
            npts_nc = rootgrp.createVariable(f'{var_name}_npts', 'i8', dimensions=("ref_alt"))
            nprfs_nc = rootgrp.createVariable(f'{var_name}_nprfs', 'i8', dimensions=("ref_alt"))

            # Fill the data
            val_nc[:] = biglambda_ms[PRF_VAL]
            uc_nc[:] = biglambda_ms.loc[:, ('ucr', 'ucs', 'uct', 'ucu')].pow(2).sum(axis=1).pow(0.5)
            npts_nc[:] = biglambda_ms['n_pts']
            nprfs_nc[:] = biglambda_ms['n_prfs']

            # Set the Variable attributes
            setattr(val_nc, 'long_name', f'Big Lambda profile of {var_name}')
            setattr(val_nc, 'units', prfs.var_info[PRF_VAL]['prm_unit'])

            setattr(val_nc, 'long_name',
                    f'Total uncertainty of the Big Lambda profile of {var_name} (k=1)')
            setattr(val_nc, 'units', prfs.var_info[PRF_VAL]['prm_unit'])

            setattr(npts_nc, 'long_name', 'Number of individual data point combined in each bin')
            setattr(npts_nc, 'units', '')

            setattr(nprfs_nc, 'long_name', 'Number of distinct delta profiles combined in each bin')
            setattr(nprfs_nc, 'units', '')

            # TODO: deal with the region chunks too

        rootgrp.close()