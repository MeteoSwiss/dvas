"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Error management

"""


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


class ConfigGenMaxLenError(ConfigReadError):
    """Exception class for max length config generator error"""


class ConfigNodeError(ConfigError):
    """Error in config node"""


class ConfigGetError(ConfigError):
    """Error in get config value"""


class ConfigLabelNameError(ConfigGetError):
    """Error in config label name"""


class ExprInterpreterError(ConfigError):
    """Error in expression interpreter"""


class NonTerminalExprInterpreterError(ExprInterpreterError):
    """Error in non terminal expression interpreter"""


class TerminalExprInterpreterError(ExprInterpreterError):
    """Error in terminal expression interpreter"""


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
