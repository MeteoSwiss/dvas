"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Error management

"""

# TODO
#  Modify exceptions using "raise MyCustomExc from my_raised_exc".

class dvasError(Exception):
    """General error class for dvas."""

    ERR_MSG = 'dvas Error'

    def __str__(self):
        return f"{super().__str__()}\n\n{'*' * 5}{self.ERR_MSG}{'*' * 5}"


class LogDirError(dvasError):
    """Exception for error in creating log directory"""


class DBIOError(dvasError):
    """Exception for dvas database IOError"""


class ProfileError(dvasError):
    """Exception for dvas profile error"""


class LoadError(dvasError):
    """Exception for dvas load error"""
