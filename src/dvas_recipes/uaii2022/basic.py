"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: basic high-level recipes for the UAII2022 campaign
"""

# Import general Python packages
import logging
import pandas as pd

# Import dvas modules and classes
# from dvas.logger import recipes_logger as logger
from dvas.environ import path_var
from dvas.logger import log_func_call
from dvas.data.data import MultiRSProfile, MultiGDPProfile
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, TAG_CLN_NAME, FLG_DESCENT_NAME
from dvas.hardcoded import TAG_GDP_NAME
from dvas.dvas import Database as DB

# Import from dvas_recipes
from .. import dynamic
from ..recipe import for_each_flight, for_each_var
from ..errors import DvasRecipesError
from . import tools
from .. import utils as dru

logger = logging.getLogger(__name__)


@log_func_call(logger, time_it=True)
def prf_summary():
    """ Exports a summary of the different profiles in the DB. """

    view = DB.extract_global_view()

    # Save this file to csv
    fn_out = path_var.output_path / (dynamic.CURRENT_STEP_ID + '_profile_list.csv')
    view.to_csv(fn_out, index=False)
    logger.info('Created profile list: %s', fn_out)


@log_func_call(logger, time_it=False)
def flag_descent(prfs):
    """ Set a dedicated flag for any point beyond the burst point.

    Note:
        This function simply uses the metadata info to set the flags. It indirectly assumes that
        the profiles have not been shifted in any way (yet).

    Args:
        prfs (MultiRSProfile|MultiGDPProfile): the profiles to flag (individually).

    Returns:
        MultiRSProfile|MultiGDPProfile: the flagged profiles.

    """

    # Loop through each profile, and figure out if I need to flag anything
    for prf in prfs:

        # Begin with some sanity checks
        if 'bpt_time' not in prf.info.metadata.keys():
            logger.error("'bpt_time' not found in metadata for: %s", prf.info.src)
            logger.error("Descent data could not be flagged !")
            continue
        if prf.info.metadata['bpt_time'] is None:
            logger.warning('"bpt_time" is not set for: %s', prf.info.src)
            continue

        # Extract the burst point, and try to convert it into a time delta
        bpt_time = (prf.info.metadata['bpt_time']).split(' ')
        if len(bpt_time) != 2:
            raise DvasRecipesError('Ouch ! bpt_time is weird: %s' % (prf.info.metadata['bpt_time']))

        bpt_time = pd.Timedelta(float(bpt_time[0]), bpt_time[1])

        # Set the flag for anything beyond the burst point
        which = prf.data.index.get_level_values('tdt') >= bpt_time
        prf.set_flg(FLG_DESCENT_NAME, True, index=which)

    return prfs


@log_func_call(logger, time_it=False)
def cleanup_steps(prfs, resampling_freq, crop_descent):
    """ Execute a series of cleanup-steps common to GDP and non-GDP profiles. This function is here
    to avoid duplicating code. The cleanup-up profiles are directly saved to the DB with the tag:
    TAG_CLN_NAME

    Args:
        prfs (MultiRSProfile|MultiGDPProfile): the profiles to cleanup.
        resampling_freq (str): time step frequency, to feed :py:func:`pandas.timedelta_range`, e.g.
            '1s'.
        crop_descent (bool): if True, and data with the flag "descent" will be cropped out for good.

    """

    # Flag descent data (do this *after* the resampling so I do not need to worry about it)
    prfs = flag_descent(prfs)

    # Crop the descent data if warranted
    if crop_descent:
        for (ind, prf) in enumerate(prfs):
            prfs[ind].data = prf.data.loc[prf.has_flg(FLG_DESCENT_NAME) == 0]

    # Resample the profiles as required
    prfs.resample(freq=resampling_freq, inplace=True, chunk_size=dynamic.CHUNK_SIZE,
                  n_cpus=dynamic.N_CPUS)

    # Save back to the DB
    prfs.save_to_db(
        add_tags=[TAG_CLN_NAME, dynamic.CURRENT_STEP_ID],
        rm_tags=dru.rsid_tags(pop=dynamic.CURRENT_STEP_ID)
        )


@for_each_var
@for_each_flight
@log_func_call(logger, time_it=True)
def cleanup(start_with_tags, **args):
    """ Highest-level function responsible for doing an initial cleanup of the data.

    Args:
        start_with_tags (str|list): list of tags to identify profiles to clean in the db.
        **args: arguments to be fed to :py:func:`.cleanup_steps`.

    """

    # Format the tags
    tags = dru.format_tags(start_with_tags)

    # Extract the flight info
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    filt = tools.get_query_filter(tags_in=tags + [eid, rid],
                                  tags_out=dru.rsid_tags(pop=tags))

    # Let's extract the summary of what the DB contains
    db_view = DB.extract_global_view()

    # I need to treat GDPs and non-GDPs separately, since the former have uncertainties that also
    # need to be cleaned accordingly.

    # Start with the GDPs
    if db_view.is_gdp[(db_view.rid == rid) & (db_view.eid == eid)].any():
        logger.info('Cleaning GDP profiles for flight %s and variable %s',
                    dynamic.CURRENT_FLIGHT,
                    dynamic.CURRENT_VAR)

        gdp_prfs = MultiGDPProfile()
        gdp_prfs.load_from_db(f'and_({filt}, tags("{TAG_GDP_NAME}"))', dynamic.CURRENT_VAR,
                              tdt_abbr=dynamic.INDEXES[PRF_REF_TDT_NAME],
                              alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME],
                              ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                              ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                              uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                              ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                              inplace=True)

        logger.info('Loaded %i GDP profiles from the DB.', len(gdp_prfs))

        cleanup_steps(gdp_prfs, **args)

    # Process the non-GDPs, if any
    if not db_view.is_gdp[(db_view.rid == rid) & (db_view.eid == eid)].all():
        logger.info('Cleaning non-GDP profiles for flight %s and variable %s',
                    dynamic.CURRENT_FLIGHT,
                    dynamic.CURRENT_VAR)

        # Extract the data from the db
        rs_prfs = MultiRSProfile()
        rs_prfs.load_from_db(f'and_({filt}, not_(tags("{TAG_GDP_NAME}")))',
                             dynamic.CURRENT_VAR, dynamic.INDEXES[PRF_REF_TDT_NAME],
                             alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME])

        logger.info('Loaded %i RS profiles from the DB.', len(rs_prfs))

        cleanup_steps(rs_prfs, **args)
