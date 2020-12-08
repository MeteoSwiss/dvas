"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Required attributes definition for
.config.ConfigManager.Tag class.

"""

# Import current packages modules
from ...database.model import Tag

# Define
TAG_NONE = ''
TAG_RAW_VAL = 'raw'
TAG_GDP_VAL = 'gdp'
TAG_DERIVED_VAL = 'derived'
TAG_EMPTY_VAL = 'empty'

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{Tag.tag_txt.name}$": {
        "type": "string",
    },
    rf"^{Tag.tag_desc.name}$": {
        "type": "string"
    },
}

#: list: Constant node values
CONST_NODES = [
    {
        Tag.tag_txt.name: TAG_NONE,
        Tag.tag_desc.name: 'None'
    },
    {
        Tag.tag_txt.name: TAG_RAW_VAL,
        Tag.tag_desc.name: 'Data are raw'
    },
    {
        Tag.tag_txt.name: TAG_GDP_VAL,
        Tag.tag_desc.name: 'GRUAN Data Product'
    },
    {
        Tag.tag_txt.name: TAG_DERIVED_VAL,
        Tag.tag_desc.name: 'Data are derived from raw'
    },
    {
        Tag.tag_txt.name: TAG_EMPTY_VAL,
        Tag.tag_desc.name: 'Data are empty'
    },
    {
        Tag.tag_txt.name: 'resampled',
        Tag.tag_desc.name: 'Data are resampled'
    },
    {
        Tag.tag_txt.name: 'sync',
        Tag.tag_desc.name: 'Data are synchronized'
    },
]

#: str: Config manager key name
KEY = Tag.__name__

#: str: Node name able to be generated by regexp
NODE_GEN = Tag.tag_txt.name
