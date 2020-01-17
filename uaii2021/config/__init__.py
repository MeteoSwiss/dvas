
# Import python packages and modules
import re

X_CHAR = 'x'
RAW_DATA_CONFIG_PARAM = [X_CHAR, 'idx', 'T', 'RH', 'P', 'A', 'WF', 'WD']
RAW_DATA_CONFIG_PARAM_NO_X = [arg for arg in RAW_DATA_CONFIG_PARAM if arg is not X_CHAR]
CONFIG_NAN_EQ = -999999

# Define
CONST_KEY_NM = 'const'
CONST_KEY_PATTERN = re.compile(rf'{CONST_KEY_NM}')
CONFIG_ITEM_PATTERN = {
    key: re.compile(val) for key, val in {
        'ref_dt': r"(\d{4})([0-1]\d)([0-1]\d)([0-2]\d)([0-6]\d)",
        'param': r"(T)|(RH)|(P)|(A)|(WF)|(WD)",
        'ms': r"ms_\d{1,5}",
        'type': r"type_\w+",
        'batch': r"batch_\d{1}",
        'flight': r"flight_\d{1,2}",
        'instr': r"instr_\w+"
    }.items()
}

