"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: dvas dynamic variables

"""

#: bool: whether the DB should be stored in memory, or not.
# Placing this here for now ... probably there's a better place for it.
DB_IN_MEMORY = False

#: int: size in which Profiles get sliced when combining them, to speed up computing. Use
# the dvas_optimize entry point to find the best values for this, depending on the machine.
CHUNK_SIZE = 150
