"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Required attributes definition for
.config.ConfigManager.OrigData class.

"""

# Import from current packages modules
from ...hardcoded import MODEL_PAT, PRM_AND_FLG_PRM_PAT
from ...database.model import Info as TableInfo
from ...database.model import MetaData as TableMetaData
from ...database.model import Tag as TableTag
from ...database.model import Object as TableObject

# Define global field name
EDT_FLD_NM = TableInfo.edt.name  # Datetime field name
TAG_FLD_NM = TableTag.__name__.lower() + 's'  # Tag field name
META_FLD_NM = TableMetaData.__name__.lower()  # Metadata field name

VALUE_FLD_NM = 'value'  # Value column field name

# Define csv field name
CSV_USE_DEFAULT_FLD_NM = 'csv_use_default'
CSV_DELIMITER_FLD_NM = 'csv_delimiter'
CSV_HEADER_FLD_NM = 'csv_header'
CSV_INDEX_COL_FLD_NM = 'csv_index_col'
CSV_SKIPINITSPACE_FLD_NM = 'csv_skipinitialspace'
CSV_SKIPROWS_FLD_NM = 'csv_skiprows'
CSV_SKIP_BLANK_LINES_FLD_NM = 'csv_skip_blank_lines'
CSV_DELIM_WHITESPACE_FLD_NM = 'csv_delim_whitespace'
CSV_COMMENT_FLD_NM = 'csv_comment'
CSV_NA_VALUES_FLD_NM = 'csv_na_values'
CSV_SKIPFOOTER_FLD_NM = 'csv_skipfooter'
CSV_ENCODING_FLD_NM = 'csv_encoding'

#: list: Fields keys passed to expression interpreter
EXPR_FIELD_KEYS = [
    EDT_FLD_NM, TableObject.srn.name,
    TableObject.pid.name, TAG_FLD_NM, META_FLD_NM
]

#: list: Node pattern
NODE_PATTERN = [MODEL_PAT, PRM_AND_FLG_PRM_PAT]

#: dict: Default values of labels
LABEL_VAL_DEF = {
    TAG_FLD_NM: [],
    META_FLD_NM: {},
    CSV_USE_DEFAULT_FLD_NM: False,
    CSV_DELIMITER_FLD_NM: ';',
    CSV_HEADER_FLD_NM: 'infer',
    CSV_INDEX_COL_FLD_NM: None,
    CSV_SKIPINITSPACE_FLD_NM: False,
    CSV_SKIPROWS_FLD_NM: 0,
    CSV_SKIP_BLANK_LINES_FLD_NM: True,
    CSV_DELIM_WHITESPACE_FLD_NM: False,
    CSV_COMMENT_FLD_NM: '#',
    CSV_NA_VALUES_FLD_NM: None,
    CSV_SKIPFOOTER_FLD_NM: 0,
    CSV_ENCODING_FLD_NM: 'utf_8',
}

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{EDT_FLD_NM}$": {
        "type": "string",
    },
    rf"^{TableObject.srn.name}$": {
        "type": "string",
    },
    rf"^{TableObject.pid.name}$": {
        "type": "string",
    },
    rf"^{TAG_FLD_NM}$": {
        "type": 'array',
        "items": {
            "type": "string",
        },
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
                            {"type": 'null'},
                            {"type": 'string'},
                            {"type": 'number'},
                        ]
                    }
                },
                "additionalProperties": False,
            },
        ]
    },
    rf"^{VALUE_FLD_NM}$": {
        "type": "string"
    },
    rf"^{CSV_USE_DEFAULT_FLD_NM}$": {
        "type": "boolean"
    },
    rf"^{CSV_DELIMITER_FLD_NM}$": {
        'anyOf': [
            {"type": "null"},
            {"type": 'string'}
        ]
    },
    rf"^{CSV_HEADER_FLD_NM}$": {
        "oneOf": [
            {"type": "null"},
            {
                "type": "integer",
                'minimum': 0
            },
            {
                "const": "infer",
            }
        ]
    },
    rf"^{CSV_INDEX_COL_FLD_NM}$": {
        "oneOf": [
          {
              "type": "null",
          },
          {
              "const": False,
          }
        ]
    },
    rf"^{CSV_SKIPINITSPACE_FLD_NM}$": {
        "type": "boolean"
    },
    rf"^{CSV_SKIPROWS_FLD_NM}$": {
        "oneOf": [
          {
              "type": "integer",
              'minimum': 0,
          },
          {
              "type": 'string'
          }
        ]
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
        'anyOf': [
            {"type": "null"},
            {"type": 'string'},
            {"type": 'array', "items": {"type": "string"}, "minItems": 1, "uniqueItems": True}
        ]
    },
    rf"^{CSV_ENCODING_FLD_NM}$": {
        'anyOf': [
            {"type": "null"},
            {"type": 'string'}
        ]
    },
}

#: str: Config manager key name
KEY = 'OrigData'
