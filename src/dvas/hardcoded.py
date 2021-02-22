"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: hardcoded parameters and variables for dvas.

"""

from pathlib import Path

# Define
RE_UPPER_W = r'[A-Z0-9]'
RE_LOWER_W = r'[a-z0-9]'


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

#: str: GDP file extension
GDP_FILE_EXT = 'nc'

#: list of str: Data file extension
CSV_FILE_EXT = ['csv', 'txt']

#: list of str: Flag data file extension
FLAG_FILE_EXT = ['flg']

#: list of str: Config file extension
CONFIG_FILE_EXT = ['yml', 'yaml']

#: str: Event id regexp pattern
EID_PAT = r'^e:\w+$'

#: str: Rig id regexp pattern
RID_PAT = r'^r:\w+$'

#: str: Name of the integer index for the pandas DataFrame of Profiles, RSProfile, GDPProfiles
PRF_REF_INDEX_NAME = '_idx'

#: str: Suffix used for flag parameter's name or file's name
FLAG_PRM_NAME_SUFFIX = '_flag'

#: str: Suffix used for flag parameter's desc
FLAG_PRM_DESC_PREFIX = 'Flag of '


# Models
# -------

#: str: Instrument type pattern (e.g. VAI-RS92, MET_LAB-C50, RS92-GDP_002)
MODEL_PAT = rf"{RE_UPPER_W}+(({RE_UPPER_W})|([\-\_]))*{RE_UPPER_W}"

#: str: CSV file model catching group pattern (e.g RS41.PAY_20171024T120000)
CSV_FILE_MDL_PAT = r"^(" + MODEL_PAT + r")\.[\w\-]+\."

#: str: GDP file model catching group pattern (e.g PAY-RS-01_2_RS41-GDP-BETA_001_20170712T000000_1-002-001.nc)
GDP_FILE_MDL_PAT = r"^[A-Z]{3}\-[A-Z]{2}\-\d{2}\_\d\_([\w\-]+\_\d{3})\_\d{8}T"

# Parameters
# ----------

#: str: Parameter pattern (e.g. tre200s0, uorpros1, uorprosu_r)
PRM_PAT = rf"{RE_LOWER_W}+(({RE_LOWER_W})|([\_]))*(?:(?<!{FLAG_PRM_NAME_SUFFIX}))"

#: str: Flag parameter pattern (e.g. tre200s0, tre200s0_flag, uorpros1, uorpros1_flag)
FLAG_PRM_PAT = rf"{RE_LOWER_W}+(({RE_LOWER_W})|([\_]))*(?:(?<={FLAG_PRM_NAME_SUFFIX}))"

#: str: Parameter and flag parameter pattern (e.g. tre200s0, tre200s0_flag, uorpros1, uorpros1_flag)
PRM_AND_FLAG_PRM_PAT = rf"(?:(({PRM_PAT})|({FLAG_PRM_PAT})))"

#: str: Name of the time delta index for the pandas DataFrame of RSProfile, GDPProfiles
PRF_REF_TDT_NAME = 'tdt'

#: str: Name of the altitude index for the pandas DataFrame of Profiles, RSProfile, GDPProfiles
PRF_REF_ALT_NAME = 'alt'

#: str: Name of the variable column for the pandas DataFrame of Profiles, RSProfile, GDPProfiles
PRF_REF_VAL_NAME = 'val'

#: str: Name of the Rig-correlated uncertainty column for the pandas DataFrame of GDPProfiles
PRF_REF_UCR_NAME = 'ucr'

#: str: Name of the Spatial-correlated uncertainty column for the pandas DataFrame of GDPProfiles
PRF_REF_UCS_NAME = 'ucs'

#: str: Name of the Temporal-correlated uncertainty column for the pandas DataFrame of GDPProfiles
PRF_REF_UCT_NAME = 'uct'

#: str: Name of the uncorrelated uncertainty column for the pandas DataFrame of GDPProfiles
PRF_REF_UCU_NAME = 'ucu'

<<<<<<< HEAD
=======
#: str: Name of the flag column for the pandas DataFrame of Profiles, RSProfile, GDPProfiles
PRF_REF_FLG_NAME = 'flg'

>>>>>>> Implement new hardcoded names

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


# Flags
# -----

#: str: Flag's name for raw NA values
TAG_EMPTY_NAME = 'raw_na'

#: str: Flag's desc for raw NA values
TAG_EMPTY_DESC = 'Raw NA values'

#: str: Flag's name for resampled values
TAG_INVALID_NAME = 'invalid'

#: str: Flag's desc for resampled values
TAG_INVALID_DESC = 'Invalid values'

#: str: Flag's name for interpolated values
TAG_INTERP_NAME = 'interp'

#: str: Flag's desc for interpolated values
TAG_INTERP_DESC = "Interpolated values"
