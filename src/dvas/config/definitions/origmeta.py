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
from ...database.model import EventsInstrumentsParameters as eip
from ...database.model import InstrType, Instrument

# Define default root parameters
NODE_PARAMS_DEF = {
    eip.event_id.name: None,
}

# Define parameter JSON_SCHEMA
PARAMETER_PATTERN_PROP = {
    rf"^{eip.event_dt.name}$": {
        "type": 'string',
    },
    rf"^{Instrument.instr_id.name}$": {
        "type": 'string',
        "pattern": INSTR_PAT
    },
    rf"^{eip.event_id.name}$": {
        "type": 'string',
        "pattern": EVENT_PAT
    },
    rf"^{eip.batch_id.name}$": {
        "type": 'string',
        "pattern": BATCH_PAT
    },
    rf"^{eip.day_event.name}$": {
        "type": "boolean",
    },
}
