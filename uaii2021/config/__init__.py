
X_CHAR = 'x'
RAW_DATA_CONFIG_PARAM = [X_CHAR, 'idx', 'T', 'RH', 'P', 'A', 'WF', 'WD']
RAW_DATA_CONFIG_PARAM_NO_X = [arg for arg in RAW_DATA_CONFIG_PARAM if arg is not X_CHAR]
CONFIG_NAN_EQ = -999999

# Define
PARAM_KEY_NM = 'param'
NODE_PAT_DICT = {
    'instr_type': r"instr_type_\w+",
    'batch': r"batch_\d{1}",
    'flight': r"flight_\d{2}",
    'instr': r"instr_\w+",
    PARAM_KEY_NM: r'{}'.format(PARAM_KEY_NM)
}

