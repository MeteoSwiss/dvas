"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Package facade

"""

# Import from current package
from .database.database import DatabaseManager
from .data.io import update_db
from .data.data import MultiProfile as DataMultiProfile
from .data.data import MultiRSProfile as DataMultiRSProfile
from .data.data import MultiGDPProfile as DataMultiGDPProfile
from .database.explorer import DatabasePrompt
from .database.search import SearchInfoExpr
from .logger import LogManager


class Log:
    """Facade class for user interactions with log"""

    @staticmethod
    def start_log(mode=1, level='INFO'):
        """ Starts the dvas logging.

        Args:
            mode (int): Log output mode.
                0 (No log) | 1 (Log to file only) | 2 (Log to file + console) |
                3 (Log to console only). Defaults to 1.
            level (str): Log level.
                Allowed: 'DEBUG'|'D'|'INFO'|'I'|'WARNING'|'WARN'|'W'|'ERROR'|'E'.
                Defaults to 'INFO'
        """

        log_inst = LogManager(mode, level)
        log_inst.init_log()

    @staticmethod
    def stop_log():
        """ Stop logging"""
        LogManager.clear_log()


class Database:
    """Facade class for user interactions with database."""

    @staticmethod
    def init():
        """ Initialize the dvas DB.

        """
        DatabaseManager()

    @staticmethod
    def refresh_db():
        """ Refreshes the database, by deleting the existing tables and reloading them from the
        existing metadata.
        """
        DatabaseManager().refresh_db()

    @staticmethod
    def fetch_original_data(search_prms, strict=True):
        """Fetch new original data and save to DB

        Args:
            search_prms (str | list of str): Search parameter value.
            strict (bool, optional): Search strict. Defaults to True.

        """

        # Set search prm list
        if isinstance(search_prms, str):
            search_prms = list(search_prms)

        # Update DB
        for search_prm in search_prms:
            update_db(search_prm, strict=strict)

    @staticmethod
    def explore():
        """Explore DB method"""
        prmt = DatabasePrompt()
        prmt.cmdloop()

    @staticmethod
    def extract_global_view():
        """Extract global DB view

        Return:
            pd.DataFrame

        """

        return SearchInfoExpr.extract_global_view()


class MultiProfile(DataMultiProfile):
    """Facade class for user interactions with MultiProfile"""


class MultiRSProfile(DataMultiRSProfile):
    """Facade class for user interactions with MultiRSProfile"""


class MultiGDPProfile(DataMultiGDPProfile):
    """Facade class for user interactions with MultiGDPProfile"""
