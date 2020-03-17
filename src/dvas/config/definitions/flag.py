"""
This module contains the required attributes definition for
.config.ConfigManager.InstrType which is a .config.ConfigManager child.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import current packages modules
from ...database.model import Flag

# Define parameter JSON_SCHEMA for user input
PARAMETER_PATTERN_PROP = {
    rf"^{Flag.bit_number.name}$": {
        "type": "integer",
        "minimum": 1,
    },
    rf"^{Flag.desc.name}$": {
        "type": "string"
    }
}

# DEFINE constant node values
CONST_NODES = [
    {
        Flag.bit_number.name: 0,
        Flag.desc.name: 'Raw data'
    }
]
