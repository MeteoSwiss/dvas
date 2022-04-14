"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Required attributes definition for
.config.ConfigManager.OrigMeta class.

"""

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    r"^\w+$": {
        'oneOf': [
            {"type": 'null'},
            {"type": 'string'},
            {"type": 'number'},
            {"type": 'boolean'},
        ]
    },
}

#: str: Config manager key name
KEY = 'CSVOrigMeta'
