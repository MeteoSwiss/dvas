"""
Copyright(c) 2020 MeteoSwiss, contributors listed in AUTHORS

Distributed under the terms of the BSD 3 - Clause License.

SPDX - License - Identifier: BSD - 3 - Clause

Module contents: Required attributes definition for
.config.ConfigManager.GDPData class.

"""

# Import from current packages modules
from ..pattern import INSTR_TYPE_PAT, PARAM_PAT
from ...database.model import Data, Instrument, EventsInfo

# Define
EVENT_DT_FIELD_NM = 'dt_field'
SN_FIELD_NM = 'sn_field'
TAG_FIELD_NM = 'tag_field'
TIME_NM = 'time_field'
PARAM_NM = 'param_field'
LAMBDA_NM = 'lambda'

INDEX_NM = Data.index.name
VALUE_NM = Data.value.name

#: list: Node pattern
NODE_PATTERN = [INSTR_TYPE_PAT, PARAM_PAT]

#: dict: Node parameters default value
NODE_PARAMS_DEF = {
    EVENT_DT_FIELD_NM: '',
    SN_FIELD_NM: '',
    TAG_FIELD_NM: [],
    TIME_NM: '',
    PARAM_NM: '',
    LAMBDA_NM: 'lambda x: x',
}

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{EVENT_DT_FIELD_NM}$": {
        "type": "string",
    },
    rf"^{SN_FIELD_NM}$": {
        "type": "string",
    },
    rf"^{TAG_FIELD_NM}$": {
        "type": 'array',
        "items": {
            "type": "string",
        },
        "minItems": 1,
        "uniqueItems": True
    },
    rf"^{TIME_NM}$": {
        "type": "string",
    },
    rf"^{PARAM_NM}$": {
        "type": "string",
    },
    rf"^{LAMBDA_NM}$": {
        "type": 'string',
        "pattern": r"^\s*lambda\s*\w+\s*\:.+"
    },
}

#: str: Config manager key name
KEY = 'GDPData'
