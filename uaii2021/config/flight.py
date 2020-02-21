
# Import
from . import X_CHAR, ID_NAME
from . import RAW_DATA_CONFIG_PARAM
from . import RAW_DATA_CONFIG_PARAM_NO_X

# Define node order
NODE_PATTERN = [r'flight_\d{1,2}$']

# Define default root parameters
ROOT_PARAMS_DEF = {
    'id': '',
    'day_flight': True,
    'launch_dt': '',
    'batch_amount': 1
}

# Define parameter JSON_SCHEMA
PARAMETER_PATTERN_PROP = {
    r"^id$": {
        "type": "string",
        "pattern": r"(^f((\d)|([1-9]\d{1,4}))$)|(^$)",
    },
    r"^day_flight$": {
        "type": "boolean"
    },
    r"^launch_dt$": {
        "type": "string",
    },
    r"^batch_amount$": {
        "type": "integer",
        'minimum': 1,
        'maximum': 3,
    }
}
