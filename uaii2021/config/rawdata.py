
# Import
from . import RAW_DATA_CONFIG_PARAM
from . import RAW_DATA_CONFIG_PARAM_NO_X
from . import CONST_KEY_NM

# Define node order
NODE_ORDER = ['type', 'instr', 'flight', 'batch']

# Define default root parameters
ROOT_PARAMS_DEF = {
    CONST_KEY_NM: {
        'idx_unit': 'ms',
        'dt_format': None,
        'delimiter': ';',
        'usecols': list(range(len(RAW_DATA_CONFIG_PARAM_NO_X))),
        'namecols': RAW_DATA_CONFIG_PARAM_NO_X,
        'x_dec': -3,
        'x_a': 1.0,
        'x_b': 0.0,
        'type_name': None,
        'skiprows': 0
    }
}

# Define parameter JSON_SCHEMA
PARAMETER_SCHEMA = {
    "type": "object",
    "patternProperties": {
        r"^idx_unit$": {
            "type": "string",
            "enum": ['dt', 'ms', 'meters']
        },
        r"^dt_format$": {
            'anyOf': [
                {"type": "null"},
                {"type": 'string'}
            ]
        },
        r"^delimiter$": {"type": 'string'},
        r"^usecols$": {
            "type": 'array',
            "items": {
                "type": "integer",
                "minimum": 0,
            },
            "minItems": 1,
            "uniqueItems": True
        },
        r"^namecols$": {
            "type": 'array',
            "items": {
                "type": "string",
                'enum': RAW_DATA_CONFIG_PARAM_NO_X
            },
            "minItems": 1,
            "uniqueItems": True
        },
        r"^({})_dec$".format('|'.join(RAW_DATA_CONFIG_PARAM)): {
            "type": 'integer',
            "maximum": 3,
            "minimum": -4
        },
        r"^({})_a$".format('|'.join(RAW_DATA_CONFIG_PARAM)): {
            "type": 'number'
        },
        r"^({})_b$".format('|'.join(RAW_DATA_CONFIG_PARAM)): {
            "type": 'number'
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
        }
    },
    "additionalProperties": False
}
