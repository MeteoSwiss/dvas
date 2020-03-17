"""
This module contains the required attributes definition for
.config.ConfigManager.Instrument which is a .config.ConfigManager child.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import from current packages modules
from ..pattern import INSTR_TYPE_KEY, INSTR_TYPE_PAT
from ..pattern import INSTR_KEY, INSTR_PAT
from ..pattern import EVENT_KEY, EVENT_PAT, BATCH_KEY, BATCH_PAT
from ..pattern import PROFILEDT_KEY, MS_KEY

# Define parameter JSON_SCHEMA
PARAMETER_PATTERN_PROP = {
    rf"^{INSTR_TYPE_KEY}$": {
        "type": 'string',
        "pattern": INSTR_TYPE_PAT
    },
    rf"^{INSTR_KEY}$": {
        "type": 'string',
        "pattern": INSTR_PAT
    },
    rf"^{EVENT_KEY}$": {
        "type": 'string',
        "pattern": EVENT_PAT
    },
    rf"^{BATCH_KEY}$": {
        "type": 'string',
        "pattern": BATCH_PAT
    },
    rf"^{PROFILEDT_KEY}$": {
        "type": "string",
    },
    rf"^{MS_KEY}$": {
        "type": "string",
    },
}
