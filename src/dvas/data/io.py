"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: IO management

"""

# Import from external packages

# Import from current package
from .linker import LocalDBLinker, CSVHandler, GDPHandler
from ..database.database import DatabaseManager
from ..database.model import Parameter
from ..logger import localdb, rawcsv
from ..environ import path_var


def update_db(search, strict=False):
    """Update database.

    Args:
        search (str): prm_abbr search criteria.
        strict (bool, optional): If False, match for any sub-string.
            If True match for entire string. Default to False.

    .. uml::

        @startuml
        hide footbox

        update_db -> CSVHandler: handle(file_path, prm_abbr)
        activate CSVHandler

        CSVHandler -> GDPHandler: handle(file_path, prm_abbr)
        activate GDPHandler

        CSVHandler <- GDPHandler: data
        deactivate  GDPHandler

        update_db <- CSVHandler: data
        deactivate   CSVHandler

        @enduml

    """

    # Init linkers
    db_mngr = DatabaseManager()
    db_linker = LocalDBLinker()

    # Define chain of responsibility for loadgin from raw
    handler = CSVHandler()
    handler.set_next(GDPHandler())

    # Search prm_abbr
    if strict is True:
        search_dict = {'where': Parameter.prm_abbr == search}
    else:
        search_dict = {'where': Parameter.prm_abbr.contains(search)}

    prm_abbr_list = [
        arg[0] for arg in db_mngr.get_or_none(
            Parameter,
            search=search_dict,
            attr=[[Parameter.prm_abbr.name]],
            get_first=False
        )
    ]

    # If no matching parameters were found, issue a warning and stop here.
    if len(prm_abbr_list) == 0:
        localdb.info("No database parameter found for the query: %s", search)
        return None

    # Log
    localdb.info("Update db for following parameters: %s", prm_abbr_list)

    # Scan path
    origdata_path_scan = list(path_var.orig_data_path.rglob("*.*"))

    # Loop loading
    for prm_abbr in prm_abbr_list:

        # Log
        rawcsv.info("Start reading files for '%s'", prm_abbr)

        # Scan files
        new_orig_data = []
        for file_path in origdata_path_scan:
            result = handler.handle(file_path, prm_abbr)
            if result:
                new_orig_data.append(result)
                # Log
                rawcsv.info(
                    "Files '%s' was treated", file_path
                )
            else:
                # Log
                rawcsv.debug(
                    "Files '%s' was left untouched", file_path
                )

        # Log
        rawcsv.info("Finish reading files for '%s'", prm_abbr)
        rawcsv.info(
            "Found %d new data while reading files for '%s'",
            len(new_orig_data),
            prm_abbr
        )

        # Log
        localdb.info(
            "Start inserting in local DB new found data for '%s'", prm_abbr
        )

        # Save to DB
        db_linker.save(new_orig_data)

        # Log
        localdb.info(
            "Finish inserting in local DB new found data for '%s'", prm_abbr
        )
