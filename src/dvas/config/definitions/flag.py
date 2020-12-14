"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Required attributes definition for `.config.ConfigManager.Flag class`.

"""

# Import current packages modules
from ...database.model import Flag

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{Flag.bit_number.name}$": {
        "type": "integer",
        "minimum": 10,
    },
    rf"^{Flag.flag_abbr.name}$": {
        "type": "string"
    },
    rf"^{Flag.flag_desc.name}$": {
        "type": "string"
    }
}

#: list: Constant node values
CONST_NODES = [
    {
        Flag.bit_number.name: 0,
        Flag.flag_abbr.name: 'raw_na',
        Flag.flag_desc.name: "Raw NA data"
    },
    {
        Flag.bit_number.name: 1,
        Flag.flag_abbr.name: 'resampled',
        Flag.flag_desc.name: "Resampled data"
    },
    {
        Flag.bit_number.name: 2,
        Flag.flag_abbr.name: 'interp',
        Flag.flag_desc.name: "Interpolated data"
    },
    {
        Flag.bit_number.name: 3,
        Flag.flag_abbr.name: 'day',
        Flag.flag_desc.name: "Day measurement point"
    },
    {
        Flag.bit_number.name: 4,
        Flag.flag_abbr.name: 'night',
        Flag.flag_desc.name: "Night measurement point"
    },
]

#: str: Config manager key name
KEY = Flag.__name__

#: str: Node name able to be generated by regexp
NODE_GEN = ''
