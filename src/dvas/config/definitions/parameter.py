"""
This module contains the required attributes definition for class
dvas.config.config.Parameter.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import current packages modules
from ..pattern import PARAM_PAT
from ...database.model import Parameter

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{Parameter.prm_abbr.name}$": {
        "type": "string",
        "pattern": PARAM_PAT,
    },
    rf"^{Parameter.prm_desc.name}$": {
        "type": "string"
    }
}
