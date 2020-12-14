"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Regex patterns used into the package.

"""

# Import python packages and modules
import re

# Define
RE_UPPER_W = r'[A-Z0-9]'
RE_LOWER_W = r'[a-z0-9]'

#: str: Instrument type pattern (e.g. VAI-RS92, MET_LAB-C50, RS92-GDP_002)
INSTR_TYPE_PAT = rf"{RE_UPPER_W}+(({RE_UPPER_W})|([\-\_]))*{RE_UPPER_W}"
INSTR_TYPE_RE = re.compile(INSTR_TYPE_PAT)

#: str: Parameter pattern (e.g. tre200s0, uorpros1, uorprosu_r)
PARAM_PAT = rf"{RE_LOWER_W}+(({RE_LOWER_W})|([\_]))*{RE_LOWER_W}"
param_re = re.compile(PARAM_PAT)
