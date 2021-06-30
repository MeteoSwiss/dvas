# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains GRUAN-related routines, including correlation rules for GDP uncertainties.

"""

# Import from Python packages
import functools
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd

# Import from current package
from ...logger import log_func_call
from ...logger import tools_logger as logger
from ...errors import DvasError
from ...hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME, PRF_REF_FLG_NAME
from ...hardcoded import PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME
from ..tools import df_to_chunks
from .utils import process_chunk
from ...data.data import MultiGDPProfile
from ...data.strategy.data import GDPProfile
from ...database.database import InfoManager

@log_func_call(logger)
def combine(gdp_prfs, binning=1, method='weighted mean', mask_flgs=None, chunk_size=150, n_cpus=1):
    ''' Combines and (possibly) rebins GDP profiles, with full error propagation.

    Note:

        This function requires profiles that have been resampled (if applicable) and synchronized
        beforehand. This implies that the `_idx` index must be identical for all Profiles.

    Args:
        gdp_profs (dvas.data.data.MultiGDPProfile): synchronized GDP profiles to combine.
        binning (int, optional): the number of profile steps to put into a bin. Defaults to 1.
        method (str, optional): combination rule. Can be one of
            ['weighted mean', 'mean', or 'delta']. Defaults to 'weighted mean'.
        mask_flgs (str|list of str, optional): (list of) flag(s) to ignore when combining profiles.
        chunk_size (int, optional): to speed up computation, Profiles get broken up in chunks of
            that length. The larger the chunks, the larger the memory requirements. The smaller the
            chunks the more items to process. Defaults to 150.
        n_cpus (int|str, optional): number of cpus to use. Can be a number, or 'max'. Set to 1 to
            disable multiprocessing. Defaults to 1.

    Returns:
        (dvas.data.data.MultiGDPProfile): the combined GDP profile.

    '''

    # Some safety checks first of all
    if not isinstance(binning, (int, np.integer)):
        raise DvasError('Ouch! binning must be of type int, not %s' % (type(binning)))
    if binning <= 0:
        raise DvasError('Ouch! binning must be greater or equal to 1 !')
    if method not in ['weighted mean', 'mean', 'delta']:
        raise DvasError('Ouch! Method %s unsupported.' % (method))

    if not isinstance(chunk_size, (int, np.integer)):
        raise DvasError('Ouch! chunk_size should be an int, not {}'.format(type(chunk_size)))

    if not isinstance(n_cpus, (int, np.integer)):
        if n_cpus == 'max':
            n_cpus = mp.cpu_count()
        else:
            raise DvasError('Ouch! n_cpus should be an int, not {}'.format(type(n_cpus)))

    # Make sure I am not asking for more cpus than available
    if n_cpus > mp.cpu_count():
        logger.warning('% cpus were requested, but I only found %i.', n_cpus, mp.cpu_count())
        n_cpus = mp.cpu_count()

    # Check that all the profiles belong to the same event and the same rig. Anything else
    # doesn't make sense.
    if len(set(gdp_prfs.get_info('eid'))) > 1 or \
       len(set(gdp_prfs.get_info('rid'))) > 1:
        raise DvasError('Ouch ! I will only combine GDPs that are from the same event+rig combo.')

    # Have all the profiles been synchronized ? Just trigger a warning for now. Maybe users simply
    # did not add the proper tag.
    if any(['sync' not in item for item in gdp_prfs.get_info('tags')]):
        logger.warning('No "sync" tag found. Is this intended ?')

    # How many gdps do we have ?
    n_prf = len(gdp_prfs)

    # How long are the profiles ?
    len_gdps = {len(prf) for prf in gdp_prfs}

    # Trigger an error if they do not have the same lengths.
    if len(len_gdps) > 1:
        raise DvasError('Ouch ! GDPs must have the same length to be combined. '+
                        'Have these been synchronized ?')

    # Turn the set back into an int.
    len_gdps = len_gdps.pop()

    # For a delta, I can only have two profiles
    if method == 'delta' and n_prf != 2:
        raise DvasError('Ouch! I can only make a delta between 2 GDPs, not %i !' % (n_prf))

    # Fine tune the chunk size to be sure that no bins is being split up if the binning is > 1.
    if binning > 1:
        chunk_size += chunk_size % binning
        logger.info("Adjusting the chunk size to %i, given the binning of %i.", chunk_size, binning)

    # Let's get started for real
    # First, let's extract all the information I (may) need, i.e. the values, errors, and total
    # errors.
    x_dx = gdp_prfs.get_prms([PRF_REF_ALT_NAME, PRF_REF_TDT_NAME, PRF_REF_VAL_NAME,
                              PRF_REF_FLG_NAME, PRF_REF_UCR_NAME, PRF_REF_UCS_NAME,
                              PRF_REF_UCT_NAME, PRF_REF_UCU_NAME, 'uc_tot'],
                              mask_flgs=mask_flgs)

    # I also need to extract some of the metadata required for computing cross-correlations.
    # Let's add it to the common DataFrame so I can carry it all in one go.
    for metadata in ['oid', 'mid', 'eid', 'rid']:
        vals = gdp_prfs.get_info(metadata)

        #Loop through it and assign the values where appropriate
        for (prf_id, val) in enumerate(vals):

            # If I am being given a list, make sure it has only 1 element. Else complain about it.
            if isinstance(val, list):
                if len(val) > 1:
                    raise DvasError("Ouch! {} ". format(metadata) +
                                    "for profile #{} ".format(prf_id) +
                                    " contains more than one value ( {} ).".format(val) +
                                    " I am too dumb to handle this. So I give up here.")

                val = val[0]

            # Actually assign the value to each measurement of the profile.
            x_dx.loc[:, (prf_id, metadata)] = val

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
        # This should definitely be fixed.
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

    # Re-assemble all the chunks into one DataFrame.
    proc_chunk = pd.concat(proc_chunks, axis=0)

    # Set the intiial flags of the combined profile.
    # Should I actually be setting any ? E.g. to make the difference between True NaN's and
    # Compatibility NaN's ? Can I even do that in here ?
    proc_chunk.loc[:, 'flg'] = 0

    # Almost there. Now we just need to package this into a clean MultiGDPProfile
    # Let's first prepare the info dict
    new_rig_tag = 'r:'+','.join([item.split(':')[1]
                                 for item in np.unique(gdp_prfs.get_info('rid')).tolist()])
    new_evt_tag = 'e:'+','.join([item.split(':')[1]
                                 for item in np.unique(gdp_prfs.get_info('eid')).tolist()])

    new_info = InfoManager(np.unique(gdp_prfs.get_info('edt'))[0], # dt
                           np.unique(gdp_prfs.get_info('oid')).tolist(),  # oids
                           tags=[new_rig_tag, new_evt_tag],
                           src='dvas combine() [{}]'.format(Path(__file__).name))

    # Let's create a dedicated Profile for the combined profile.
    # It's no different from a GDP, from the perspective of the errors.
    new_prf = GDPProfile(new_info, data=proc_chunk)

    # And finally, package this into a MultiGDPProfile entity
    out = MultiGDPProfile()
    out.update(gdp_prfs.db_variables, data=[new_prf])

    return out
