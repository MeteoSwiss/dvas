"""
This module contains the required attributes definition for class
dvas.config.config.InstrType

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import current packages modules
from ..pattern import INSTR_TYPE_PAT
from ...database.model import InstrType

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{InstrType.type_name.name}$": {
        "type": "string",
        "pattern": INSTR_TYPE_PAT,
    },
    rf"^{InstrType.desc.name}$": {
        "type": "string"
    }
}
