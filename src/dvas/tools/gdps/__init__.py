# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This sub-package contains GRUAN-related tools.

"""

# WARNING: The following line would be nice, but ABSOLUTELY cannot be enable. Doing so would result
# in a circular import. This is related to the resample strategy that requires access to corfoefs.
# TODO: should corcoefs be placed somewhere else ? I sort of like it here ...
# Simplify the import of the main GRUAN routines
#from .gdps import *
