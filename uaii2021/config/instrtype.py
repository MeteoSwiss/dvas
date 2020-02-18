
# Import
from . import X_CHAR, ID_NAME
from . import RAW_DATA_CONFIG_PARAM
from . import RAW_DATA_CONFIG_PARAM_NO_X

# Define node order
NODE_ORDER = ['instr_type']

# Define default root parameters
ROOT_PARAMS_DEF = {
    'name': '',
    'desc': '',
}

# Define parameter JSON_SCHEMA
PARAMETER_SCHEMA = {
    "type": "object",
    "patternProperties": {
        r"^name$": {
            "type": "string",
            "enum": ['dt', 's', 'ms', 'meters']
        },
        r"^dt_format$": {
            'anyOf': [
                {"type": "null"},
                {"type": 'string'}
            ]
        },
        r"^delimiter$": {
            'anyOf': [
                {"type": "null"},
                {"type": 'string'}
            ]
        },
        r"^index_col$": {
            "type": "string",
            "enum": [ID_NAME]},
        r"^header$": {
            'anyOf': [
                {"type": "null"},
                {"type": 'integer'}
            ]
        },
        r"^usecols$": {
            "type": 'array',
            "items": {
                "type": "integer",
                "minimum": 0,
            },
            "minItems": 1,
            "uniqueItems": True
        },
        r"^names$": {
            "type": 'array',
            "items": {
                "type": "string",
                'enum': RAW_DATA_CONFIG_PARAM_NO_X
            },
            "minItems": 1,
            "uniqueItems": True
        },
        rf"^({'|'.join(RAW_DATA_CONFIG_PARAM)})_func$": {
            "type": 'string',
            "pattern": r"^\s*lambda\s*\w+\s*\:.+"
        },
        r"^type_name$": {
            'anyOf': [
                {"type": "null"},
                {"type": 'string'}
            ]
        },
        r"^skiprows$": {
            'anyOf': [
                {"type": "integer", 'minimum': 0},
                {"type": 'string'}
            ]
        },
        r"^skip_blank_lines$": {"type": "boolean"},
        r"^delim_whitespace$": {"type": "boolean"}

    },
    "additionalProperties": False
}
