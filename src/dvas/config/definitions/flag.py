"""
This module contains the required attributes definition for
.config.ConfigManager.InstrType which is a .config.ConfigManager child.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import current packages modules
from ...database.model import Flag

# Define
RAWNA_ABBR = 'raw_na'
RESMPL_ABBR = 're_smpl'
UPSMPL_ABBR = 'up_smpl'
INTERP_ABBR = 'interp'
SYNC_ABBR = 'sync'
AUTOQC_ABBR = 'auto_qc'

# Define parameter JSON_SCHEMA for user input
PARAMETER_PATTERN_PROP = {
    rf"^{Flag.bit_number.name}$": {
        "type": "integer",
        "minimum": 6,
    },
    rf"^{Flag.flag_abbr.name}$": {
        "type": "string"
    },
    rf"^{Flag.desc.name}$": {
        "type": "string"
    }
}

# DEFINE constant node values
CONST_NODES = [
    {
        Flag.bit_number.name: 0,
        Flag.flag_abbr.name: RAWNA_ABBR,
        Flag.desc.name: "Raw NA"
    },
    {
        Flag.bit_number.name: 1,
        Flag.flag_abbr.name: RESMPL_ABBR,
        Flag.desc.name: "Resampled"
    },
    {
        Flag.bit_number.name: 2,
        Flag.flag_abbr.name: UPSMPL_ABBR,
        Flag.desc.name: "Resampled"
    },
    {
        Flag.bit_number.name: 3,
        Flag.flag_abbr.name: INTERP_ABBR,
        Flag.desc.name: 'Interpolated'
    },
    {
        Flag.bit_number.name: 4,
        Flag.flag_abbr.name: SYNC_ABBR,
        Flag.desc.name: 'Synchronized'
    },
{
        Flag.bit_number.name: 5,
        Flag.flag_abbr.name: AUTOQC_ABBR,
        Flag.desc.name: 'Auto QC flagged'
    },
]
