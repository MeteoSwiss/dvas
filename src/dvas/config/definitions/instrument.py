"""
This module contains the required attributes definition for
.config.ConfigManager.Instrument which is a .config.ConfigManager child.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import current packages modules
from ..pattern import INSTR_NODE_PAT, INSTR_PAT, INSTR_TYPE_PAT
from ...database.model import Instrument

# Define node order
NODE_PATTERN = [INSTR_NODE_PAT]

# Define default root parameters
ROOT_PARAMS_DEF = {
    Instrument.instr_id.name: '',
    Instrument.instr_type.name: '',
    Instrument.sn.name: '',
    Instrument.remark.name: ''
}

# Define parameter JSON_SCHEMA
PARAMETER_PATTERN_PROP = {
    rf"^{Instrument.instr_id.name}$": {
        "type": "string",
        "pattern": INSTR_PAT,
    },
    rf"^{Instrument.instr_type.name}$": {
        "type": "string",
        "pattern": INSTR_TYPE_PAT,
    },
    rf"^{Instrument.sn.name}$": {
        "type": "string"
    },
    rf"^{Instrument.remark.name}$": {
        "type": "string"
    }
}
