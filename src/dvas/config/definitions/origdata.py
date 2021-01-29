"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Required attributes definition for
.config.ConfigManager.OrigData class.

"""

# Import from current packages modules
from ..pattern import INSTR_TYPE_PAT, PARAM_PAT
from ...database.model import Data

# Define global field name
EVT_DT_FLD_NM = 'dt_field'  # Datetime field name
SRN_FLD_NM = 'srn_field'  # Serial number field name
PDT_FLD_NM = 'pdt_field'  # Product field name
TAG_FLD_NM = 'tag_field'  # Tag field name
META_FLD_NM = 'meta_field'  # Metadata field name

INDEX_FLD_NM = 'index_col'  # Index column field name
PARAM_FLD_NM = 'value_col'  # Value column field name

UNIT_FLD_NM = 'unit'  # Index unit field name

LAMBDA_FLD_NM = 'lambda'  # Lambda field name

# Define csv field name
CSV_DELIMITER_FLD_NM = 'csv_delimiter'
CSV_HEADER_FLD_NM = 'csv_header'
CSV_INDEX_COL_FLD_NM = 'csv_index_col'
CSV_NAMES_FLD_NM = 'csv_names'
CSV_SKIPINITSPACE_FLD_NM = 'csv_skipinitialspace'
CSV_SKIPROWS_FLD_NM = 'csv_skiprows'
CSV_SKIP_BLANK_LINES_FLD_NM = 'csv_skip_blank_lines'
CSV_DELIM_WHITESPACE_FLD_NM = 'csv_delim_whitespace'
CSV_COMMENT_FLD_NM = 'csv_comment'
CSV_NA_VALUES_FLD_NM = 'csv_na_values'
CSV_SKIPFOOTER_FLD_NM = 'csv_skipfooter'

INDEX_NM = Data.index.name
VALUE_NM = Data.value.name

#: list: Fields keys passed to expression interpreter
EXPR_FIELD_KEYS = [
    EVT_DT_FLD_NM, SRN_FLD_NM,
    PDT_FLD_NM, TAG_FLD_NM, META_FLD_NM
]

#: list: Node pattern
NODE_PATTERN = [INSTR_TYPE_PAT, PARAM_PAT]

#: dict: Node parameters default value
NODE_PARAMS_DEF = {
    TAG_FLD_NM: [],
    META_FLD_NM: {},
    UNIT_FLD_NM: '1',
    LAMBDA_FLD_NM: 'lambda x: x',
    CSV_DELIMITER_FLD_NM: ';',
    CSV_SKIPINITSPACE_FLD_NM: False,
    CSV_SKIPROWS_FLD_NM: 0,
    CSV_SKIP_BLANK_LINES_FLD_NM: True,
    CSV_DELIM_WHITESPACE_FLD_NM: False,
    CSV_COMMENT_FLD_NM: '#',
    CSV_NA_VALUES_FLD_NM: ['/'],
    CSV_SKIPFOOTER_FLD_NM: 0
}

#: dict: Constant nodes
CONST_NODES = {
    CSV_HEADER_FLD_NM: 0,
    CSV_NAMES_FLD_NM: [VALUE_NM],
}

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{EVT_DT_FLD_NM}$": {
        "type": "string",
    },
    rf"^{SRN_FLD_NM}$": {
        "type": "string",
    },
    rf"^{PDT_FLD_NM}$": {
        "type": "string",
    },
    rf"^{TAG_FLD_NM}$": {
        "type": 'array',
        "items": {
            "type": "string",
        },
        "minItems": 1,
        "uniqueItems": True
    },
    rf"^{META_FLD_NM}$": {
        "oneOf": [
            {"type": 'null'},
            {
                "type": 'object',
                "patternProperties": {
                    r"^[\w\.]+$": {
                        'oneOf': [
                            {"type": 'string'},
                            {"type": 'number'},
                        ]
                    }
                },
                "additionalProperties": False,
            },
        ]
    },
    rf"^{PARAM_FLD_NM}$": {
        "oneOf": [
            {
                "type": "integer",
                "minimum": 0
            },
            {
                "type": "string",
            }
        ]
    },
    rf"^{UNIT_FLD_NM}$": {
        "type": "string",
    },
    rf"^{LAMBDA_FLD_NM}$": {
        "type": 'string',
        "pattern": r"^\s*lambda\s*\w+\s*\:.+"
    },
    rf"^{CSV_DELIMITER_FLD_NM}$": {
        'anyOf': [
            {"type": "null"},
            {"type": 'string'}
        ]
    },
    rf"^{CSV_SKIPINITSPACE_FLD_NM}$": {
        "type": "boolean"
    },
    rf"^{CSV_SKIPROWS_FLD_NM}$": {
        "type": "integer",
        'minimum': 0,
    },
    rf"^{CSV_SKIPFOOTER_FLD_NM}$": {
        "type": "integer",
        'minimum': 0,
    },
    rf"^{CSV_SKIP_BLANK_LINES_FLD_NM}$": {
        "type": "boolean"
    },
    rf"^{CSV_DELIM_WHITESPACE_FLD_NM}$": {
        "type": "boolean"
    },
    rf"^{CSV_COMMENT_FLD_NM}$": {
        "type": "string",
        "enum": ['#']
    },
    rf"^{CSV_NA_VALUES_FLD_NM}$": {
        "type": 'array',
        "items": {
            "type": "string"
        },
        "minItems": 1,
        "uniqueItems": True
    },
}

#: str: Config manager key name
KEY = 'OrigData'
