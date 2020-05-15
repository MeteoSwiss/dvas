"""
This module contains the required attributes definition for class
dvas.config.config.Instrument.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import current packages modules
from ..pattern import INSTR_PAT, INSTR_TYPE_PAT
from ...database.model import Instrument

#: dict: Node parameters default value
NODE_PARAMS_DEF = {
    Instrument.sn.name: '',
    Instrument.remark.name: ''
}

#: dict: Parameter pattern properties (JSON_SCHEMA)
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
