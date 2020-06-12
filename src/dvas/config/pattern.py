"""
This module contains the regex patterns used into the package.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import python packages and modules
import re

#: str: Instrument type pattern (e.g. vai-rs92, met_lab-c50)
INSTR_TYPE_PAT = r"\w{3,8}\-\w{3,8}"
INSTR_TYPE_RE = re.compile(INSTR_TYPE_PAT)

#: str: Instrument key name
INSTR_PREFIX = 'i'

#: str: Instrument pattern (e.g. i1, i10, i203)
INSTR_PAT = rf'{INSTR_PREFIX}([1-9](\d(\d)?)?)'
instr_re = re.compile(INSTR_PAT)

#: str: Parameter pattern (e.g. tre200s0, uorpros1)
PARAM_PAT = r"\w{8}"
param_re = re.compile(PARAM_PAT)
