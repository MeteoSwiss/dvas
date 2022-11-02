"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: IO management

"""

# Import from external packages
import logging

# Import from current package
from .linker import LocalDBLinker
from .linker import CSVHandler, GDPHandler
from .linker import FlgCSVHandler, FlgGDPHandler
from ..config.config import OrigData
from ..database.database import DatabaseManager
from ..database.model import Prm as TableParameter
from ..environ import path_var
from ..errors import DvasError

# Create logger
logger = logging.getLogger(__name__)


def update_db(search, strict=False):
    """Update database.

    Args:
        search (str): Parameter name search criteria.
        strict (bool, optional): If False, match for any sub-string.
            If True match for entire string. Default to False.

    .. uml::

        @startuml
        hide footbox

        update_db -> CSVHandler: handle(file_path, prm_name)
        activate CSVHandler

        CSVHandler -> GDPHandler: handle(file_path, prm_name)
        activate GDPHandler

        CSVHandler <- GDPHandler: data
        deactivate  GDPHandler

        update_db <- CSVHandler: data
        deactivate   CSVHandler

        @enduml

    """

    # If search is None, there is no need to go much further
    if search is None:
        logger.debug("I was asked to search for None ... so that's what I'll return.")
        return None

    # Init orig data config
    origdata_config_mngr = OrigData()
    origdata_config_mngr.read()

    # Init linkers
    db_mngr = DatabaseManager()
    db_linker = LocalDBLinker()

    # Define chain of responsibility for loading from original
    handler = CSVHandler(origdata_config_mngr)
    handler.\
        set_next(GDPHandler(origdata_config_mngr)).\
        set_next(FlgCSVHandler(origdata_config_mngr)).\
        set_next(FlgGDPHandler(origdata_config_mngr))

    # Search prm_name
    if strict is True:
        search_dict = {'where': TableParameter.prm_name == search}
    else:
        search_dict = {'where': TableParameter.prm_name.contains(search)}

    prm_name_list = [
        arg[0] for arg in db_mngr.get_or_none(
            TableParameter,
            search=search_dict,
            attr=[[TableParameter.prm_name.name]],
            get_first=False
        )
    ]

    # If no matching parameters were found, issue a warning and stop here.
    if len(prm_name_list) == 0:
        logger.warning("No database parameter found for the query: %s", search)
        return None

    # Log
    logger.info("Update db for following parameters: %s", prm_name_list)

    # Test
    if path_var.orig_data_path is None:
        raise DvasError('Ouch ! path_var.orig_data_path is None')

    # Scan data path (entirely)
    origdata_path_scan = list(path_var.orig_data_path.rglob("*.*"))

    # Loop loading
    for prm_name in prm_name_list:

        # Log
        logger.debug("Start reading files for '%s'", prm_name)

        # Scan files
        new_orig_data = []

        for file_path in handler.filter_files(origdata_path_scan, prm_name):
            result = handler.handle(file_path, prm_name)
            if result:
                new_orig_data.append(result)
                # Log
                logger.debug("File '%s' was treated", file_path)
            else:
                # Log
                logger.debug("File '%s' was left untouched", file_path)

        # Log
        logger.debug("Finish reading files for '%s'", prm_name)
        logger.debug("Found %d new data while reading files for '%s'", len(new_orig_data), prm_name)

        # Log
        logger.debug("Start inserting in local DB new found data for '%s'", prm_name)

        # Save to DB
        db_linker.save(new_orig_data)

        # Log
        logger.debug("Finish inserting in local DB new found data for '%s'", prm_name)

    # Delete
    del db_mngr, db_linker, handler
