# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains GRUAN-related routines, including correlation rules for GDP uncertainties.

"""

# Import from Python packages
import logging
import functools
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd

# Import from current package
from ...logger import log_func_call
from ...errors import DvasError
from ...hardcoded import PRF_TDT, PRF_ALT, PRF_VAL, PRF_FLG, PRF_UCS, PRF_UCT, PRF_UCU
from ...hardcoded import TOD_VALS, TAG_ORIGINAL, TAG_CLN, TAG_1S, TAG_SYNC
from ..tools import df_to_chunks
from ..chunks import process_chunk
from ...data.data import MultiCWSProfile
from ...data.strategy.data import CWSProfile
from ...database.database import InfoManager

# Setup local logger
logger = logging.getLogger(__name__)


@log_func_call(logger)
def combine(gdp_prfs, binning=1, method='weighted arithmetic mean',
            mask_flgs=None, chunk_size=150, n_cpus=1):
    ''' Combines and (possibly) rebins GDP profiles, with full error propagation.

    Args:
        gdp_profs (dvas.data.data.MultiGDPProfile): synchronized GDP profiles to combine.
        binning (int, optional): the number of profile steps to put into a bin. Defaults to 1.
        method (str, optional): combination rule. Can be one of
            ['weighted arithmetic mean', 'arithmetic mean', weighted circular mean',
            'circular mean', or 'delta']. Defaults to 'weighted arithmetic mean'.
        mask_flgs (str|list of str, optional): (list of) flag(s) to ignore when combining profiles.
        chunk_size (int, optional): to speed up computation, Profiles get broken up in chunks of
            that length. The larger the chunks, the larger the memory requirements. The smaller the
            chunks the more items to process. Defaults to 150.
        n_cpus (int|str, optional): number of cpus to use. Can be a number, or 'max'. Set to 1 to
            disable multiprocessing. Defaults to 1.

    Returns:
        (dvas.data.data.MultiCWSProfile, dict): the combined working standard profile, and the
            a dictionnary with the full covariance matrices for the different uncertainty types.

    Note:
        This function requires profiles that have been resampled (if applicable) and synchronized
        beforehand. This implies that the `_idx` index must be identical for all Profiles.



    '''

    # Some safety checks first of all
    if not isinstance(binning, (int, np.integer)):
        raise DvasError(f'binning must be of type int, not {type(binning)}')
    if binning <= 0:
        raise DvasError('binning must be greater or equal to 1 !')
    if method not in ['weighted arithmetic mean', 'arithmetic mean', 'weighted circular mean',
                      'circular mean', 'arithmetic delta', 'circular delta']:
        raise DvasError(f'Method {method} unsupported.')

    if not isinstance(chunk_size, (int, np.integer)):
        raise DvasError(f'chunk_size should be an int, not {type(chunk_size)}')

    if not isinstance(n_cpus, (int, np.integer)):
        if n_cpus == 'max':
            n_cpus = mp.cpu_count()
        else:
            raise DvasError(f'n_cpus should be an int, not {type(n_cpus)}')

    # Make sure I am not asking for more cpus than available
    if n_cpus > mp.cpu_count():
        logger.warning('%i cpus were requested, but I only found %i.', n_cpus, mp.cpu_count())
        n_cpus = mp.cpu_count()

    # Check that all the profiles belong to the same event and the same rig. Anything else
    # doesn't make sense.
    if len(set(gdp_prfs.get_info('eid'))) > 1 or \
       len(set(gdp_prfs.get_info('rid'))) > 1:
        raise DvasError('I will only combine GDPs that are from the same event+rig combo.')

    # Have all the profiles been synchronized ? Just trigger a warning for now. Maybe users simply
    # did not add the proper tag.
    if any(TAG_SYNC not in item for item in gdp_prfs.get_info('tags')):
        logger.warning('No "%s" tag found. Is this intended ?', TAG_SYNC)

    # How many gdps do we have ?
    n_prf = len(gdp_prfs)

    # How long are the profiles ?
    len_gdps = {len(prf) for prf in gdp_prfs}

    # Trigger an error if they do not have the same lengths.
    if len(len_gdps) > 1:
        raise DvasError('GDPs must have the same length to be combined. ' +
                        'Have these been synchronized ?')

    # Turn the set back into an int.
    len_gdps = len_gdps.pop()

    # For a delta, I can only have two profiles
    if method == 'delta' and n_prf != 2:
        raise DvasError(f'I can only make a delta between 2 GDPs, not {n_prf} !')

    # Make sure that the chunk_size is not smaller than the binning, else I cannot actually
    # bin the data as required
    if chunk_size < binning:
        chunk_size = binning
        logger.info("Adjusting the chunk size to %i, to match the requested binning level",
                    chunk_size)

    # Fine tune the chunk size to be sure that no bins is being split up if the binning is > 1.
    # 2021-09-16: fix bug #166 by *subtracting* chunk_size % binning (rather than adding it) !
    if binning > 1:
        chunk_size -= chunk_size % binning
        if chunk_size % binning > 0:
            logger.info("Adjusting the chunk size to %i, given the binning of %i.",
                        chunk_size, binning)

    # Let's get started for real
    # First, let's extract all the information I (may) need, i.e. the values, errors, and total
    # errors.
    x_dx = gdp_prfs.get_prms([PRF_ALT, PRF_TDT, PRF_VAL, PRF_FLG, PRF_UCS, PRF_UCT,
                              PRF_UCU, 'uc_tot'],
                             mask_flgs=mask_flgs, with_metadata=['oid', 'mid', 'eid', 'rid'])

    # To drastically reduce memory requirements and speed up the code significantly,
    # we will break the profiles into smaller chunks. In doing so, we avoid having to deal with
    # correlation matrices with O(10^8) elements (the majority of which are 0 anyway).
    # The idea of breaking into chunks is made possible by the fact that only neighboring levels are
    # combined with each others (so no large scale correlation factors must be dealt with).
    chunks = df_to_chunks(x_dx, chunk_size)

    merge_func = functools.partial(process_chunk, binning=binning, method=method)
    if n_cpus == 1:
        proc_chunks = map(merge_func, chunks)
    else:

        # TODO: this is a bug that I do not understand. When running the script from a ipython
        # session multiple times in a row, the __spec__ variable is only defined the first time
        # around. Having this set to anything (=None when run interactively) is crucial to the
        # multiprocessing Pool routine.
        # So for now, use a terribly dangerous workaround that I do not understand.
        # This should definitely be fixed. Or not ?
        # See #121
        import sys
        try:
            sys.modules['__main__'].__spec__ is None
        except AttributeError:
            logger.warning('BUG: __spec__ is not set. Fixing it by hand ...')
            sys.modules['__main__'].__spec__ = None

        pool = mp.Pool(processes=n_cpus)
        proc_chunks = pool.map(merge_func, chunks)
        pool.close()
        pool.join()

    # Extract the profile chunks, in order to stich them up together.
    x_ms = [item[0] for item in proc_chunks]

    # Re-assemble all the chunks into one DataFrame.
    x_ms = pd.concat(x_ms, axis=0)

    # Almost there. Now we just need to package this into a clean MultiCWSProfile
    # Let's first prepare the info dict
    new_rig_tag = 'r:'+','.join([item.split(':')[1]
                                 for item in np.unique(gdp_prfs.get_info('rid')).tolist()])
    new_evt_tag = 'e:'+','.join([item.split(':')[1]
                                 for item in np.unique(gdp_prfs.get_info('eid')).tolist()])
    one_or_more_tags = [tag for tag in list(TOD_VALS) + [TAG_1S] if any(gdp_prfs.has_tag(tag))]
    all_or_nothing_tags = [tag for tag in [TAG_ORIGINAL, TAG_CLN, TAG_SYNC]
                           if all(gdp_prfs.has_tag(tag))]

    new_fid = ','.join(set(item['fid'] if 'fid' in item.keys() else '???' for item in
                       gdp_prfs.get_info(prm='metadata')))

    new_info = InfoManager(np.unique(gdp_prfs.get_info('edt'))[0],  # dt
                           np.unique(gdp_prfs.get_info('oid')).tolist(),  # oids
                           tags=[new_rig_tag, new_evt_tag] + one_or_more_tags + all_or_nothing_tags,
                           src=f'dvas combine() [{Path(__file__).name}]')
    new_info.add_metadata('fid', new_fid)

    # Let's create a dedicated Profile for the combined profile.
    # It's no different from a GDP, from the perspective of the errors.
    new_prf = CWSProfile(new_info, data=x_ms)

    # Package this into a MultiCWSProfile entity
    out = MultiCWSProfile()
    out.update(gdp_prfs.db_variables, data=[new_prf])

    # To finish, let's piece together the covariance matrices
    # Set them up full of NaNs to start
    cov_mats = {uc_name: np.full((len(x_ms), len(x_ms)), np.nan) for uc_name in
                [PRF_UCS, PRF_UCT, PRF_UCU]}
    # Then fill them up chunk by chunk
    for item in proc_chunks:
        for uc_name in [PRF_UCS, PRF_UCT, PRF_UCU]:
            cov_mats[uc_name][item[0].index[0]:item[0].index[-1]+1,
                              item[0].index[0]:item[0].index[-1]+1] = \
                item[1][uc_name].filled(np.nan)

    return out, cov_mats
