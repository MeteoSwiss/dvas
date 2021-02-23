"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: hardcoded parameters and variables for dvas.

"""

from pathlib import Path

# Paths
# -----

#: pathlib.Path: Package path
PKG_PATH = Path(__file__).resolve(strict=True).parent

#: pathlib.Path: Matplotlib plots styles
MPL_STYLES_PATH = Path('plots/mpl_styles')

# Global var
# ----------

#: int: Config generator limit
CONFIG_GEN_LIM = 10000

#: list of str: Config file extension
CONFIG_FILE_EXT = ['yml', 'yaml']

#: str: Event id regexp pattern
EID_PAT = r'^e:\w+$'

#: str: Rig id regexp pattern
RID_PAT = r'^r:\w+$'

#: str: Name of the integer index for the pandas DataFrame of Profiles, RSProfile, GDPProfiles
PRF_REF_INDEX_NAME = '_idx'

# Tags
# ----

#: str: Tag's name for none tag in DB
TAG_NONE_NAME = ''

#: str: Tag's desc for none tag in DB
TAG_NONE_DESC = 'None'

#: str: Tag's name for raw profiles
TAG_RAW_NAME = 'raw'

#: str: Tag's desc for raw profiles
TAG_RAW_DESC = 'Raw profile'

#: str: Tag's name for GDP profiles
TAG_GDP_NAME = 'gdp'

#: str: Tag's desc for GDP profiles
TAG_GDP_DESC = 'GRUAN Data Product'

#: str: Tag's name for combined working standard profile
TAG_CWS_NAME = 'cws'

#: str: Tag's desc for combined working standard profile
TAG_CWS_DESC = 'Combined working measurement standard'

#: str: Tag's name for synchronized profile
TAG_SYNC_NAME = 'sync'

#: str: Tag's desc for synchronized profile
TAG_SYNC_DESC = 'Synchronized profile'

#: str: Tag's name for empty values in raw data
TAG_EMPTY_NAME = 'empty'

#: str: Tag's desc for empty values in raw data
TAG_EMPTY_DESC = 'Empty data'
