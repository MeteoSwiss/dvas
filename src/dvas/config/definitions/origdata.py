"""
This module contains the required attributes definition for class
dvas.config.config.OrigData

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import from current packages modules
from ..pattern import INSTR_TYPE_PAT, INSTR_PAT, PARAM_PAT
from ...database.model import Data

# Define
IDX_UNIT_NM = 'idx_unit'
DT_FORMAT_NM = 'dt_format'
LAMBDA_NM = 'lambda'
DELIMITER_NM = 'delimiter'
INDEX_COL_NM = 'index_col'
HEADER_NM = 'header'
USECOLS = 'usecols'
NAMES_NM = 'names'
DTYPE_NM = 'dtype'
SKIPROWS_NM = 'skiprows'
SKIP_BLANK_LINES_NM = 'skip_blank_lines'
DELIM_WHITESPACE_NM = 'delim_whitespace'
COMMENT_NM = 'comment'
NA_VALUES = 'na_values'

INDEX_NM = Data.index.name
VALUE_NM = Data.value.name

#: list: Node pattern
NODE_PATTERN = [INSTR_TYPE_PAT, PARAM_PAT, INSTR_PAT]

#: dict: Node parameters default value
NODE_PARAMS_DEF = {
    IDX_UNIT_NM: 'ms',
    DT_FORMAT_NM: None,
    LAMBDA_NM: 'lambda x: x',
    DELIMITER_NM: ';',
    INDEX_COL_NM: INDEX_NM,
    HEADER_NM: 'infer',
    USECOLS: [],
    NAMES_NM: [INDEX_NM, VALUE_NM],
    SKIPROWS_NM: 0,
    SKIP_BLANK_LINES_NM: True,
    DELIM_WHITESPACE_NM: False,
    COMMENT_NM: '#',
    NA_VALUES: ['/']
}

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{IDX_UNIT_NM}$": {
        "type": "string",
        "enum": ['dt', 's', 'ms', 'meters']
    },
    rf"^{DT_FORMAT_NM}$": {
        'anyOf': [
            {"type": "null"},
            {"type": 'string'}
        ]
    },
    rf"^{LAMBDA_NM}$": {
        "type": 'string',
        "pattern": r"^\s*lambda\s*\w+\s*\:.+"
    },
    rf"^{DELIMITER_NM}$": {
        'anyOf': [
            {"type": "null"},
            {"type": 'string'}
        ]
    },
    rf"^{INDEX_COL_NM}$": {
        "type": "string",
        "enum": [INDEX_NM]
    },
    rf"^{HEADER_NM}$": {
        'anyOf': [
            {
                "type": "string",
                "enum": ['infer']
            },
            {"type": 'integer'}
        ]
    },
    rf"^{USECOLS}$": {
        "type": 'array',
        "items": {
            "type": "integer",
            "minimum": 0,
        },
        "minItems": 2,
        "uniqueItems": True
    },
    rf"^{NAMES_NM}$": {
        "type": 'array',
        "items": {
            "type": "string",
            'enum': [INDEX_NM, VALUE_NM]
        },
        "minItems": 2,
        "maxItems": 2,
        "uniqueItems": True
    },
    rf"^{SKIPROWS_NM}$": {
        "type": "integer",
        'minimum': 0,
    },
    rf"^{SKIP_BLANK_LINES_NM}$": {
        "type": "boolean"
    },
    rf"^{DELIM_WHITESPACE_NM}$": {
        "type": "boolean"
    },
    rf"^{COMMENT_NM}$": {
        "type": "string",
        "enum": ['#']
    },
    rf"^{NA_VALUES}$": {
        "type": 'array',
        "items": {
            "type": "string"
        },
        "minItems": 1,
        "uniqueItems": True
    },
}
