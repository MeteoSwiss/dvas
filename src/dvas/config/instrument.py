
# Import
from . import X_CHAR, ID_NAME
from . import RAW_DATA_CONFIG_PARAM
from . import RAW_DATA_CONFIG_PARAM_NO_X

# Define node order
NODE_PATTERN = [r'instrument_\d{1,2}$']

# Define default root parameters
ROOT_PARAMS_DEF = {
    'id': '',
    'instr_type': '',
    'desc': '',
    'remark': ''
}

# Define parameter JSON_SCHEMA
PARAMETER_PATTERN_PROP = {
    r"^id$": {
        "type": "string",
        "pattern": r"(^i((\d{1})|([1-9]\d{1})|([1-9]\d{2}))$)|(^$)",
    },
    r"^instr_type$": {
        "type": "string",
        "pattern": r"(^[a-z0-9]{3,}\-[a-z0-9]+$)|($^)",
    },
    r"^desc$": {
        "type": "string"
    },
    r"^remark$": {
        "type": "string"
    }
}
