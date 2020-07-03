"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Required attributes definition for
.config.ConfigManager.OrigMeta class.

"""

# Import from current packages modules
from ...database.model import EventsInfo
from ...database.model import Instrument
from ...database.model import Tag


#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{EventsInfo.event_dt.name}$": {
        "type": 'string',
    },
    rf"^{Instrument.sn.name}$": {
        "type": 'string',
    },
    rf"^{Tag.tag_abbr.name}$": {
        "type": 'array',
        "items": {
            "type": "string",
        },
        "minItems": 1,
        "uniqueItems": True
    },
}

#: list: Constant node values
CONST_NODES = [
    {
        Tag.tag_abbr.name: 'raw',
    },
]

#: str: Config manager key name
KEY = 'OrigMeta'
