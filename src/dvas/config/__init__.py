
# Import python packages and modules
import re

from ..database.database import db_mngr, Parameter
from ..database.model import FLIGHT_PATTERN, INSTR_PATTERN, BATCH_PATTERN


param_nm = [arg[Parameter.name.name] for arg in db_mngr.get_parameter()]

X_CHAR = 'x'
ID_NAME = 'idx'
RAW_DATA_CONFIG_PARAM = [X_CHAR, ID_NAME] + param_nm
RAW_DATA_CONFIG_PARAM_NO_X = [ID_NAME] + param_nm
RAW_DATA_CONFIG_PARAM_NO_X_NO_ID = param_nm
CONFIG_NAN_EQ = -999999
ITEM_SEPARATOR = '.'


# Define
CONST_KEY_NM = 'config_values'
CONST_KEY_PATTERN = re.compile(rf'{CONST_KEY_NM}')
CONFIG_ITEM_PATTERN = {
    key: re.compile(rf"^{val}$") for key, val in {
        # Reference datetime (e.g. 2020-02-28T000000Z)
        'ref_dt': r"(\d{4})\-([0-1]\d)\-([0-3]\d)T([0-2]\d)([0-6]\d)([0-6]\d)Z",
        # Data parameter (e.g. trepros1, dklpros1)
        'param': r"[a-z0-9]{8}",
        # Meassite reference id (e.g. PAY, PAY_1, PAY_L1)
        'ms': r"[A-Z]{3}(_[A-Z0-9]+)?",
        # Instrument type reference id (e.g. vai-rs92)
        'instr_type': r"[a-z0-9]{3,}\-[a-z0-9]+",
        # Batch reference id (e.g b0, b1)
        'batch': BATCH_PATTERN,
        # Flight reference id (e.g f00, f03, f12, f34)
        'flight': FLIGHT_PATTERN,
        # Instrument reference id (e.g i0, i13, i342)
        'instr': INSTR_PATTERN,
    }.items()
}

