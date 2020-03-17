"""
This module contains the required attributes definition for
.config.ConfigManager.InstrType which is a .config.ConfigManager child.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import current packages modules
from ..pattern import INSTR_TYPE_NODE_PAT, INSTR_TYPE_PAT
from ...database.model import InstrType

# Define node order
NODE_PATTERN = [INSTR_TYPE_NODE_PAT]

# Define default root parameters
ROOT_PARAMS_DEF = {
    InstrType.name.name: '',
    InstrType.desc.name: '',
}

# Define parameter JSON_SCHEMA
PARAMETER_PATTERN_PROP = {
    rf"^{InstrType.name.name}$": {
        "type": "string",
        "pattern": INSTR_TYPE_PAT,
    },
    rf"^{InstrType.desc.name}$": {
        "type": "string"
    }
}
