"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: basic high-level recipes for the UAII2022 campaign
"""

# Import general Python packages
import logging

# Import dvas modules and classes
# from dvas.logger import recipes_logger as logger
from dvas.environ import path_var
from dvas.logger import log_func_call
from dvas.data.data import MultiRSProfile, MultiGDPProfile
from dvas.hardcoded import PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, TAG_CLN_NAME
from dvas.dvas import Database as DB

# Import from dvas_recipes
from .. import dynamic
from ..recipe import for_each_flight, for_each_var
from ..errors import DvasRecipesError
from . import tools

logger = logging.getLogger(__name__)


@log_func_call(logger, time_it=True)
def prf_summary():
    """ Exports a summary of the different profiles in the DB. """

    view = DB.extract_global_view()

    # Save this file to csv
    fn_out = path_var.output_path / (dynamic.CURRENT_STEP_ID + '_profile_list.csv')
    view.to_csv(fn_out, index=False)
    logger.info('Created profile list: %s', fn_out)


@for_each_var
@for_each_flight
@log_func_call(logger, time_it=True)
def cleanup(tags, dt):
    """ Highest-level function responsible for doing an initial cleanup of the data.

    Args:
        tags (str|list): list of tags to identify profiles to clean in the db.
        dt (int|float): time step frequency, to feed :py:func:`pandas.timedelta_range`.

    """

    # Deal with the search tags
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        raise DvasRecipesError('Ouch ! tags should be of type str|list. not: {}'.format(type(tags)))

    # Extract the flight info
    (eid, rid) = dynamic.CURRENT_FLIGHT

    # What search query will let me access the data I need ?
    filt = tools.get_query_filter(tags_in=tags+[eid, rid], tags_out=[TAG_CLN_NAME])

    # Let's extract the summary of what the DB contains
    db_view = DB.extract_global_view()

    # I need to treat GDPs and non-GDPs separately, since the former have uncertainties that also
    # need to be cleaned accordingly.

    # Start with the GDPs
    if db_view.is_gdp.any():
        logger.info('Cleaning GDP profiles for flight %s and variable %s',
                    dynamic.CURRENT_FLIGHT,
                    dynamic.CURRENT_VAR)

        gdp_prfs = MultiGDPProfile()
        gdp_prfs.load_from_db(f'and_({filt}, tags("gdp"))', dynamic.CURRENT_VAR,
                              tdt_abbr=dynamic.INDEXES[PRF_REF_TDT_NAME],
                              alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME],
                              ucr_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucr'],
                              ucs_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucs'],
                              uct_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['uct'],
                              ucu_abbr=dynamic.ALL_VARS[dynamic.CURRENT_VAR]['ucu'],
                              inplace=True)

        # Resample the profiles as required
        gdp_prfs.resample(freq=dt, inplace=True,
                          chunk_size=dynamic.CHUNK_SIZE, n_cpus=dynamic.N_CPUS)

        # Save back to the DB
        gdp_prfs.save_to_db(add_tags=[TAG_CLN_NAME])

    # Process the non-GDPs, if any
    if not db_view.is_gdp.all():
        logger.info('Cleaning non-GDP profiles for flight %s and variable %s',
                    dynamic.CURRENT_FLIGHT,
                    dynamic.CURRENT_VAR)

        # Extract the data from the db
        rs_prfs = MultiRSProfile()
        rs_prfs.load_from_db(filt, dynamic.CURRENT_VAR, dynamic.INDEXES[PRF_REF_TDT_NAME],
                             alt_abbr=dynamic.INDEXES[PRF_REF_ALT_NAME])

        rs_prfs.resample(freq=dt, inplace=True,
                         chunk_size=dynamic.CHUNK_SIZE, n_cpus=dynamic.N_CPUS)

        # Save back to the DB
        rs_prfs.save_to_db(add_tags=[TAG_CLN_NAME])
