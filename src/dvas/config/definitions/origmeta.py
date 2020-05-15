"""
This module contains the required attributes definition for class
dvas.config.config.OrigMeta

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import from current packages modules
from ..pattern import INSTR_PAT
from ..pattern import EVENT_PAT
from ..pattern import BATCH_PAT
from ...database.model import EventsInstrumentsParameters as evt_inst_prm
from ...database.model import Instrument

#: dict: Node parameters default value
NODE_PARAMS_DEF = {
    evt_inst_prm.event_id.name: None,
}

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{evt_inst_prm.event_dt.name}$": {
        "type": 'string',
    },
    rf"^{Instrument.instr_id.name}$": {
        "type": 'string',
        "pattern": INSTR_PAT
    },
    rf"^{evt_inst_prm.event_id.name}$": {
        "type": 'string',
        "pattern": EVENT_PAT
    },
    rf"^{evt_inst_prm.batch_id.name}$": {
        "type": 'string',
        "pattern": BATCH_PAT
    },
    rf"^{evt_inst_prm.day_event.name}$": {
        "type": "boolean",
    },
}
