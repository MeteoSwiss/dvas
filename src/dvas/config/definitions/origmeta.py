"""
This module contains the required attributes definition for class
dvas.config.config.OrigMeta

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import from current packages modules
from ..pattern import INSTR_PAT
from ...database.model import EventsInfo
from ...database.model import Instrument
from ...database.model import Tag


#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{EventsInfo.event_dt.name}$": {
        "type": 'string',
    },
    rf"^{Instrument.instr_id.name}$": {
        "type": 'string',
        "pattern": INSTR_PAT
    },
    rf"^{Tag.tag_abbr.name}$": {
        "type": 'array',
        "items": {
            "type": "string",
        },
        "minItems": 1,
        "uniqueItems": True
    },
}

#: str: Config manager key name
KEY = 'OrigMeta'
