"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

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
from .logger import LogManager


def init_log(mode=1, level='INFO'):
    """Init logging

    Args:
        mode (int): Log output mode. 0 (No log) | 1 (Log to file) | 2 (Log to file + console). Default to 1.
        level (str): Log level. Allowed: 'DEBUG'|'D'|'INFO'|'I'|'WARNING'|'WARN'|'W'|'ERROR'|'E'. Default to 'INFO'
    """

    log_inst = LogManager(mode, level)
    log_inst.init_log()


def clear_log():
    """Clear logging"""
    LogManager.clear_log()


class Database:
    """Facade class for user interactions for database."""

    def __init__(self, reset=False):
        """
        Args:
            reset (bool, optional): Reset DB, Defaults to False
        """
        DatabaseManager(reset_db=reset)

    @staticmethod
    def fetch_raw_data(search_prm, strict=False):
        """Fetch new raw data and save to DB

        Args:
            search_prm (str): Search parameter value.
            strict (bool, optional): Search strict. Defaults to False.

        """
        update_db(search_prm, strict=strict)


class MultiProfile(DataMultiProfile):
    """Facade class for user interactions with MultiProfile"""


class MultiRSProfile(DataMultiRSProfile):
    """Facade class for user interactions with MultiRSProfile"""


class MultiGDPProfile(DataMultiGDPProfile):
    """Facade class for user interactions with MultiGDPProfile"""
