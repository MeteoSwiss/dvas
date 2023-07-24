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
from dvas.hardcoded import PRF_ALT, PRF_VAL, PRF_UCS, PRF_UCT, PRF_UCU, PRF_FLG
from dvas.hardcoded import TAG_DTA

# Import from dvas_recipes
from ..errors import DvasRecipesError
from .. import dynamic
from . import tools
from .. import utils as dru
from .export import add_cf_attributes, set_attribute

# Setup local logger
logger = logging.getLogger(__name__)


@log_func_call(logger)
def biglambda_tod(prf_tags, mid, tods, suffix='', institution='',
                  gph_min=0, gph_bin_size=100, gph_bin_count=350):
    """ Highest-level recipe function responsible for assembling OSCAR profiles, and storing the
    result in a dedicated netCDF.

    Args:
        prf_tags (list of str): tag name(s) for the search query into the database.
        mid (str): 'mid' to process.
        tods (list of str): times-of-day to process (OR).
        suffix (str, optional): name of the netCDF file suffix. Defaults to ''.
        institution (str, optional): for the netCDF eponym field. Defaults to ''.
        gph_min (list, optional): min gph altitude to consider, in m. Defaults to 0.
        gph_bin_size (int, optional): gph bin size, in m. Defaults to 100.
        gph_bin_count (int, optional): gph bin count. Defaults to 350.

    """

    db_view = DB.extract_global_view()
    # Second sanity check - make sure the mid is in the DB
    if mid not in db_view.mid.unique().tolist():
        raise DvasRecipesError(f'mid unknown: {mid}')
    else:
        logger.info('Processing %s %s...', mid, tods)

    # Get the model name and model description
    mdesc = db_view[db_view.mid == mid].mdl_desc.unique()
    mname = db_view[db_view.mid == mid].mdl_name.unique()

    for item in [mdesc, mname]:
        assert len(item) == 1, "Too many descriptions/names for one mid"

    mdesc = mdesc[0]
    mname = mname[0]

    # Create the netCDF that will be used to store the data
    fname = '_'.join([suffix, 'big-lambda', mid, '-'.join([item[4:] for item in tods])]) + '.nc'

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

    set_attribute(rootgrp, 'd.Sonde.ModelId', f'{mid}')
    set_attribute(rootgrp, 'd.Sonde.ModelName', f'{mname}')
    set_attribute(rootgrp, 'd.Sonde.ModelDescription', f'{mdesc}')

    set_attribute(rootgrp, 'd.tod', f'{",".join([item[4:] for item in tods])}')
    set_attribute(rootgrp, 'd.biglambda.gph_min', f'{gph_min} m')
    set_attribute(rootgrp, 'd.biglambda.gph_bin_size', f'{gph_bin_size} m')

    # What search query will let me access the data I need ?
    prf_filt = tools.get_query_filter(tags_in=prf_tags + [TAG_DTA],
                                      tags_in_or=tods, tags_out=None, mids=[mid])

    # Start looking for all the variables
    for (var_name, var) in dynamic.ALL_VARS.items():

        # Skip the lat lon variables for big lambda
        if var_name in ['lat', 'lon']:
            continue

        # Load the delta profiles as Profiles (and not RSProfiles) since we're about to drop the
        prfs = MultiDeltaProfile()
        prfs.load_from_db(prf_filt, var_name,
                          alt_abbr=dynamic.INDEXES[PRF_ALT],
                          ucs_abbr=var['ucs'],
                          uct_abbr=var['uct'],
                          ucu_abbr=var['ucu'],
                          inplace=True)
        logger.info('Found %i delta_profiles for %s', len(prfs), var_name)

        # Let's extract the data I care about
        pdf = prfs.get_prms([PRF_ALT, PRF_VAL, PRF_FLG, PRF_UCS, PRF_UCT, PRF_UCU,
                            'uc_tot'], mask_flgs=None,
                            with_metadata=['oid', 'mid', 'eid', 'rid'],
                            pooled=True)

        # Sort by altitudes
        pdf.sort_values((0, PRF_ALT), inplace=True)

        # Ready to build some chunks
        chunks = [pdf.loc[(pdf.loc[:, (0, PRF_ALT)] >= item) *
                          (pdf.loc[:, (0, PRF_ALT)] < gph_bins[ind+1])].copy()
                  for (ind, item) in enumerate(gph_bins[:-1])]

        # Start processing chunks
        logger.info('Processing the high-resolution chunks for %s ...', var_name)
        proc_func = functools.partial(process_chunk, method='biglambda', return_V_mats=False)
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
        val_nc = rootgrp.createVariable(f'{var_name}_biglambda', 'f8', dimensions=("ref_alt"))
        mean_nc = rootgrp.createVariable(f'{var_name}_mean', 'f8', dimensions=("ref_alt"))
        std_nc = rootgrp.createVariable(f'{var_name}_std', 'f8', dimensions=("ref_alt"))
        uc_nc = rootgrp.createVariable(f'{var_name}_biglambda_uc', 'f8', dimensions=("ref_alt"))
        ucs_nc = rootgrp.createVariable(f'{var_name}_biglambda_ucs', 'f8', dimensions=("ref_alt"))
        uct_nc = rootgrp.createVariable(f'{var_name}_biglambda_uct', 'f8', dimensions=("ref_alt"))
        ucu_nc = rootgrp.createVariable(f'{var_name}_biglambda_ucu', 'f8', dimensions=("ref_alt"))
        npts_nc = rootgrp.createVariable(f'{var_name}_npts', 'i8', dimensions=("ref_alt"))
        nprfs_nc = rootgrp.createVariable(f'{var_name}_nprfs', 'i8', dimensions=("ref_alt"))

        # Fill the data
        val_nc[:] = biglambda_ms[PRF_VAL]
        mean_nc[:] = biglambda_ms['mean']
        std_nc[:] = biglambda_ms['std']
        uc_nc[:] = biglambda_ms.loc[:, ('ucs', 'uct', 'ucu')].pow(2).sum(axis=1).pow(0.5)
        ucs_nc[:] = biglambda_ms[PRF_UCS]
        uct_nc[:] = biglambda_ms[PRF_UCT]
        ucu_nc[:] = biglambda_ms[PRF_UCU]
        npts_nc[:] = biglambda_ms['n_pts']
        nprfs_nc[:] = biglambda_ms['n_prfs']

        # Set the Variable attributes
        setattr(val_nc, 'long_name', f'Big Lambda profile of {var_name}')
        setattr(val_nc, 'units', prfs.var_info[PRF_VAL]['prm_unit'])
        setattr(mean_nc, 'long_name', f'Mean delta profile of {var_name}')
        setattr(mean_nc, 'units', prfs.var_info[PRF_VAL]['prm_unit'])
        setattr(std_nc, 'long_name', f'Standard deviation profile of {var_name}')
        setattr(std_nc, 'units', prfs.var_info[PRF_VAL]['prm_unit'])

        setattr(uc_nc, 'long_name',
                f'Total uncertainty of the Big Lambda profile of {var_name} (k=1)')
        setattr(uc_nc, 'units', prfs.var_info[PRF_VAL]['prm_unit'])

        setattr(ucs_nc, 'long_name',
                f'Spatial-correlated uncertainty of the Big Lambda profile of {var_name} (k=1)')
        setattr(ucs_nc, 'units', prfs.var_info[PRF_VAL]['prm_unit'])

        setattr(uct_nc, 'long_name',
                'Temporal-correlated uncertainty of the Big Lambda profile of ' +
                f'{var_name} (k=1)')
        setattr(uct_nc, 'units', prfs.var_info[PRF_VAL]['prm_unit'])

        setattr(ucu_nc, 'long_name',
                f'Uncorrelated uncertainty of the Big Lambda profile of {var_name} (k=1)')
        setattr(ucu_nc, 'units', prfs.var_info[PRF_VAL]['prm_unit'])

        setattr(npts_nc, 'long_name', 'Number of individual data point combined in each bin')
        setattr(npts_nc, 'units', '')

        setattr(nprfs_nc, 'long_name', 'Number of distinct delta profiles combined in each bin')
        setattr(nprfs_nc, 'units', '')

        # Let us now deal with the region chunks
        for region in ['PBL', 'FT', 'UTLS', 'MUS']:

            logger.info('Processing the %s chunk for %s ...', region, var_name)
            flg = f"is_in_{region}"

            pdf = prfs.get_prms(
                [PRF_ALT, PRF_VAL, PRF_FLG, PRF_UCS, PRF_UCT, PRF_UCU, 'uc_tot'],
                mask_flgs=None, request_flgs=[flg, 'has_valid_cws'],
                with_metadata=['oid', 'mid', 'eid', 'rid'],
                pooled=True)

            val = process_chunk(pdf, method='biglambda')

            # Let's compute the mean altitude of all the points in the region, as a rough
            # indication of where we stand. This is variable dependant, since I get points
            # that have a valid CWS (which may not always be there for certain variables)
            set_attribute(rootgrp, f'd.{var_name}.{region}.mean_ref_alt',
                          f'{pdf[(0, PRF_ALT)].mean():.1f} {prfs.var_info[PRF_ALT]["prm_unit"]}'
                          )

            uc_val = val[0].loc[:, ('ucs', 'uct', 'ucu')].pow(2).sum(axis=1).pow(0.5)[0]

            set_attribute(rootgrp, f'd.{var_name}.{region}.biglambda',
                          f'{val[0]["val"][0]:.5f} {prfs.var_info[PRF_VAL]["prm_unit"]}')
            set_attribute(
                rootgrp, f'd.{var_name}.{region}.biglambda.uc', f"{uc_val:.5f} " +
                f'{prfs.var_info[PRF_VAL]["prm_unit"]}')
            set_attribute(rootgrp, f'd.{var_name}.{region}.biglambda.ucs',
                          f'{val[0]["ucs"][0]:.5f} {prfs.var_info[PRF_VAL]["prm_unit"]}')
            set_attribute(rootgrp, f'd.{var_name}.{region}.biglambda.uct',
                          f'{val[0]["uct"][0]:.5f} {prfs.var_info[PRF_VAL]["prm_unit"]}')
            set_attribute(rootgrp, f'd.{var_name}.{region}.biglambda.ucu',
                          f'{val[0]["ucu"][0]:.5f} {prfs.var_info[PRF_VAL]["prm_unit"]}')

            set_attribute(rootgrp, f'd.{var_name}.{region}.mean_delta',
                          f'{val[0]["mean"][0]:.5f} {prfs.var_info[PRF_VAL]["prm_unit"]}')
            set_attribute(rootgrp, f'd.{var_name}.{region}.std',
                          f'{val[0]["std"][0]:.5f} {prfs.var_info[PRF_VAL]["prm_unit"]}')

            set_attribute(rootgrp, f'd.{var_name}.{region}.biglambda.npts',
                          f'{val[0]["n_pts"][0]}')
            set_attribute(rootgrp, f'd.{var_name}.{region}.biglambda.nprfs',
                          f'{val[0]["n_prfs"][0]}')

    rootgrp.close()


@log_func_call(logger)
def compute_biglambda(start_with_tags, mids=None, tods=None, **kwargs):
    """ Highest-level recipe function responsible for assembling Big Lambda profiles, and storing
    the result in a dedicated netCDF.

    Args:
        start_with_tags (str|list of str): tag name(s) for the search query into the database.
        mids (list, optional): list of 'mid' to process. Defaults to None = all.
        tods (list of list|str, optional): list of time-of-days to process sequentially, e.g.
            ['daytime', ['nighttime', 'twilight']]. Defaults to None=all.
        **kwargs (optional): all fed to biglambda_tod().

    """

    # Cleanup the tags
    prf_tags = dru.format_tags(start_with_tags)

    # Very well, let us first extract the 'mid', if they have not been provided
    db_view = DB.extract_global_view()
    if mids is None:
        mids = db_view.mid.unique().tolist()

    # Basic sanity check of mid
    if not isinstance(mids, list):
        raise DvasRecipesError(f'I need a list of mids, not: {mids}')

    if tods is None:
        tods = [['daytime', 'nighttime', 'twilight']]
    if isinstance(tods, str):
        tods = [tods]

    # Cleanup the tods to match the dvas syntax
    tods = [[item] if isinstance(item, str) else item for item in tods]
    tods = [[f'tod:{subitem}' for subitem in item] for item in tods]

    # Start looping for the computation
    for mid in mids:
        for tod in tods:
            biglambda_tod(prf_tags, mid, tod, **kwargs)
