"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Error management

"""

import inspect

# Import form current packages/modules
from . import __name__ as main_pkg_nm


class DvasError(Exception):
    """General exception class for dvas."""


class ConfigError(DvasError):
    """Exception for error in config"""


class ConfigPathError(ConfigError):
    """Exception for error in config file path"""


class ConfigReadError(ConfigError):
    """Error while reading config"""


class ConfigReadYAMLError(ConfigReadError):
    """Exception for error in reading YAML file"""


class ConfigCheckJSONError(ConfigReadError):
    """Exception for error in checking JSON"""



class ConfigNodeError(ConfigError):
    """Error in config node"""


class ConfigItemKeyError(ConfigError):
    """Error in config key item"""


class ConfigGenMaxLenError(ConfigError):
    """Exception class for max length config generator error"""



class LogDirError(DvasError):
    """Exception for error in creating log directory"""


class DBError(DvasError):
    """Exception for dvas database error"""


class DBIOError(DBError):
    """Exception for dvas database IOError"""


class SearchError(DBError):
    """Exception for dvas database search error"""


class ProfileError(DvasError):
    """Exception for dvas profile error"""


class LoadError(DvasError):
    """Exception for dvas load error"""
