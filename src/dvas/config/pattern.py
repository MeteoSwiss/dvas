"""
This module contains the regex patterns used into the package.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import python packages and modules
import re

#: str: Instrument type key name
INSTR_TYPE_KEY = 'instr_type'
INSTR_TYPE_NODE_PAT = rf'{INSTR_TYPE_KEY}_\d+'

#: str: Instrument type pattern (e.g. vai-rs92, metlab-c50)
INSTR_TYPE_PAT = r"[a-z0-9]{3,}-[a-z0-9]+"
INSTR_TYPE_RE = re.compile(INSTR_TYPE_PAT)

#: str: Instrument key name
INSTR_KEY = 'instrument'
INSTR_NODE_PAT = rf'{INSTR_KEY}_\d+'
INSTR_PREFIX = INSTR_KEY[0]

#: str: Instrument pattern (e.g. i1, i10, i203)
INSTR_PAT = rf'{INSTR_PREFIX}([1-9]\d*)'
INSTR_RE = re.compile(INSTR_PAT)

#: str: Event key name
EVENT_KEY = 'event'
EVENT_NODE_PAT = rf'{EVENT_KEY}_\d+'
EVENT_PREFIX = EVENT_KEY[0]

#: str: Event pattern (e.g. e1, e10, e203)
EVENT_PAT = rf'{EVENT_PREFIX}[1-9]\d*'
EVENT_RE = re.compile(EVENT_PAT)

#: str: Batch key name
BATCH_KEY = 'batch'
BATCH_PREFIX = BATCH_KEY[0]

#: str: Batch pattern (e.g. b1, b20, b205)
BATCH_PAT = rf'{BATCH_PREFIX}[1-9]\d*'
BATCH_RE = re.compile(BATCH_PAT)

#: str: Parameter key name
PARAM_KEY = 'parameter'
PARAM_NODE_PAT = rf'{PARAM_KEY}_\d+'

#: str: Parameter pattern (e.g. tre200s0, uorprot1)
PARAM_PAT = r"[a-z0-9]{8}"
PARAM_RE = re.compile(PARAM_PAT)

#: str: Raw data key name
ORIGDATA_KEY = 'orig_data'

#: str: Raw meta key name
ORIGMETA_KEY = 'orig_meta'

#: str: Profile datetime stamp
PROFILEDT_KEY = 'profile_dt'

#: str: Meassite name
MS_KEY = 'meassite'

#: str: Flag name
FLAG_KEY = 'flag'
FLAG_NODE_PAT = rf'{FLAG_KEY}_([1-9]\d*)'
