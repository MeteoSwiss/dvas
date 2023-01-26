"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: dvas dynamic variables

"""

#: bool: whether the DB should be stored in memory, or not.
# Placing this here for now ... probably there's a better place for it `/`
DB_IN_MEMORY = False

#: bool: whether the profile data should be stored in the db, or in individual text file on disk.
DATA_IN_DB = True
