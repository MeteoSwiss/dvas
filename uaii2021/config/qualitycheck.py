
# Import
from . import RAW_DATA_CONFIG_PARAM
from . import CONFIG_NAN_EQ
from . import PARAM_KEY_NM

# Define node order
NODE_ORDER = ['flight', 'batch', 'instr']

# Define root parameters
ROOT_PARAMS_DEF = {
    PARAM_KEY_NM: {
        'idx': None,
        'rep_param': 'x',
        'rep_val': CONFIG_NAN_EQ
    }
}

# Define parameter JSON_SCHEMA
PARAMETER_SCHEMA = {
    "type": "object",
    "patternProperties": {
        r"^idx$": {
            'anyOf': [
                {"type": "null"},
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                        },
                        "minItems": 2,
                        "maxItems": 2,
                        "uniqueItems": True
                    },
                    "minItems": 1,
                    "uniqueItems": True
                }
            ]
        },
        r"^rep_param$": {
            "type": 'string',
            "enum": RAW_DATA_CONFIG_PARAM
        },
        r"^rep_val$": {
            "type": 'number'
        }
    },
    "additionalProperties": False
}
