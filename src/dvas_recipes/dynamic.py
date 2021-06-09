"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: dynamic variables, that get modified on the fly by the Recipes.
"""

#: list: list of event_id, rig_id pairs identifying all the flights to to be processed. E.g.:
# [[12345, 1], [12346, 1]]
ALL_FLIGHTS = None

#: list: current event_id, rig_id of the flight to be processed next. E.g.: [12345, 1]
CURRENT_FLIGHT = None

#: dict: All variables to be processed by the recipe. Should be set once only, e.g.:
# {'temp': {'ucr': 'temp_ucr', 'ucs': 'temp_ucs', 'uct': 'temp_uct', 'ucu': None}, 'rh': {...}, ...}
ALL_VARS = None

#: dict: current variable name to be processed, and associated uncertainties, e.g.:
# {'temp': {'ucr': 'temp_ucr', 'ucs': 'temp_ucs', 'uct': 'temp_uct', 'ucu': None}}
CURRENT_VAR = None

#: dict: variable database name associated to the 'tdt' and 'alt' indexes
INDEXES = None

#: int|str: current step id
CURRENT_STEP_ID = None
