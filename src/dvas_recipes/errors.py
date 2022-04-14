"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Error management

"""


class DvasRecipesError(Exception):
    """General error class for dvas_recipes."""

    ERR_MSG = 'dvas_recipes Error'

    def __str__(self):
        return f"{super().__str__()}\n\n{'*' * 5}{self.ERR_MSG}{'*' * 5}"
