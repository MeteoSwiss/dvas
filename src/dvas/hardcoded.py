"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: hardcoded parameters and variables for dvas.

"""

from pathlib import Path

# Define
RE_UPPER_W = r'[A-Z0-9]'
RE_LOWER_W = r'[a-z0-9]'
RE_UPPERLOWER_W = r'[a-zA-Z0-9]'


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
CSV_FILE_EXT = ['csv', 'txt', 'cor']

#: list of str: Flag data file extension
FLG_FILE_EXT = ['flg']

#: list of str: Config file extension
CONFIG_FILE_EXT = ['yml', 'yaml']

#: str: Event id regexp pattern
EID_PAT = r'^e:\w+$'

#: int: Len of the eid tag, comprised of 'e:' + 6 GRUAN id digits
EID_LEN = 2+6

#: str: Rig id regexp pattern
RID_PAT = r'^r:\w+$'

#: str: TimeOfDay regexp pattern
TOD_PAT = r'^tod:\w+$'

#: set: possible values for the TimeOfDay
TOD_VALS = ('tod:daytime', 'tod:nighttime', 'tod:twilight')

#: str: Suffix used for flag parameter's name or file's name
FLG_PRM_NAME_SUFFIX = '_flag'

#: str: Suffix used for flag parameter's desc
FLG_PRM_DESC_PREFIX = 'Flag of '


# Models
# -------
#: str: Instrument type pattern (e.g. VAI-RS92, MET_LAB-C50, RS92-GDP_002)
MODEL_PAT = rf"{RE_UPPERLOWER_W}+(({RE_UPPERLOWER_W})|([\-\_\(\)]))*{RE_UPPERLOWER_W}"

#: str: CSV file model catching group pattern (e.g RS41.etc..., DFM-17.etc ...)
CSV_FILE_MDL_PAT = r"^(" + MODEL_PAT + r")\.[\w\-]+\."

#: str: GDP file model catching group pattern (e.g PAY-RS-01_2_RS41-GDP-BETA_001_20170712Tetc.nc)
GDP_FILE_MDL_PAT = r"^[A-Z]{3}\-[A-Z]{2}\-\d{2}\_\d\_([\w\-]+\_\d{3})\_\d{8}T"

# Parameters
# ----------
#: str: Parameter pattern (e.g. tre200s0, uorpros1, uorprosu_r)
PRM_PAT = rf"{RE_LOWER_W}+(({RE_LOWER_W})|([\_]))*(?:(?<!{FLG_PRM_NAME_SUFFIX}))"

#: str: Flag parameter pattern (e.g. tre200s0, tre200s0_flag, uorpros1, uorpros1_flag)
FLG_PRM_PAT = rf"{RE_LOWER_W}+(({RE_LOWER_W})|([\_]))*(?:(?<={FLG_PRM_NAME_SUFFIX}))"

#: str: Parameter and flag parameter pattern (e.g. tre200s0, tre200s0_flag, uorpros1, uorpros1_flag)
PRM_AND_FLG_PRM_PAT = rf"(?:(({PRM_PAT})|({FLG_PRM_PAT})))"


# DataFrame columns
# -----------------
#: str: Name of the integer index for the pandas DataFrame of Profiles, RSProfile, GDPProfiles
PRF_IDX = '_idx'

#: str: Name of the time delta index for the pandas DataFrame of RSProfile, GDPProfiles
PRF_TDT = 'tdt'

#: str: Name of the altitude index for the pandas DataFrame of Profiles, RSProfile, GDPProfiles
PRF_ALT = 'alt'

#: str: Name of the variable column for the pandas DataFrame of Profiles, RSProfile, GDPProfiles
PRF_VAL = 'val'

#: str: Name of the Spatial-correlated uncertainty column for the pandas DataFrame of GDPProfiles
PRF_UCS = 'ucs'

#: str: Name of the Temporal-correlated uncertainty column for the pandas DataFrame of GDPProfiles
PRF_UCT = 'uct'

#: str: Name of the uncorrelated uncertainty column for the pandas DataFrame of GDPProfiles
PRF_UCU = 'ucu'

#: str: Name of the flag column for the pandas DataFrame of Profiles, RSProfile, GDPProfiles
PRF_FLG = 'flg'

# Tags
# ----
#: str: Tag name for none tag in DB
TAG_NONE = ''
#: str: Tag desc for none tag in DB
TAG_NONE_DESC = 'None'

#: str: Tag name for original profiles
TAG_ORIGINAL = 'is_original'
#: str: Tag desc for original profiles
TAG_ORIGINAL_DESC = 'Original profile unaltered by dvas'

#: str: Tag name for cleaned profiles
TAG_CLN = 'is_cleaned'
#: str: Tag desc for cleaned profiles
TAG_CLN_DESC = 'Cleaned-up profile'

#: str: Tag name for resampled profiles.
TAG_1S = 'is_resampled'
#: str: Tag  description for resampled profiles.
TAG_1S_DESC = 'Profile is resampled onto a regular time-step grid'

#: str: Tag name for synchronized profile
TAG_SYNC = 'is_synchronized'
#: str: Tag desc for synchronized profile
TAG_SYNC_DESC = 'Synchronized profile'

#: str: Tag name for GDP profiles
TAG_GDP = 'is_gdp'
#: str: Tag desc for GDP profiles
TAG_GDP_DESC = 'GRUAN Data Product'

#: str: Tag name for combined working standard profile
TAG_CWS = 'is_cws'
#: str: Tag desc for combined working standard profile
TAG_CWS_DESC = 'Combined working measurement standard'

#: str: Tag name for profile deltas with CWS
TAG_DTA = 'is_delta'
#: str: Tag desc for profile deltas with CWS
TAG_DTA_DESC = 'Profile minus CWS'

#: str: Tag name for empty original data
TAG_EMPTY = 'is_empty'
#: str: Tag desc for empty original data
TAG_EMPTY_DESC = 'Empty data'

# Flags
# -----

#: str: Flag name for when we have no original data (i.e. synced gaps, etc ...)
FLG_NOPRF = 'no_original_data'
#: str: Flag desc for no data
FLG_NOPRF_DESC = 'No original data existing for this point (i.e. out of bound)'

#: str: Flag name for invalid values
FLG_ISINVALID = 'is_invalid'
#: str: Flag desc for invalid values
FLG_ISINVALID_DESC = 'Invalid data (e.g. NaN value for real GDP measurement)'

#: str: Flag name for invalid values
FLG_WASINVALID = 'was_invalid'
#: str: Flag desc for invalid values
FLG_WASINVALID_DESC = 'Invalid data (e.g. NaN uct value for real GDP gph measurement) ' +\
                      'that was corrected following official GRUAN instructions'

#: str: Flag name for interpolated values
FLG_INTERP = 'is_interpolated'
#: str: Flag desc for interpolated values
FLG_INTERP_DESC = "Interpolated values"

#: str: Flag name for resampled values
FLG_INCOMPATIBLE = 'is_GDP_incompatible'
#: str: Flag desc for resampled values
FLG_INCOMPATIBLE_DESC = 'GDPs are incompatible with one another'

#: str: Flag name for regions with valid CWS
FLG_HASCWS = 'has_valid_cws'
#: str: Flag desc
FLG_HASCWS_DESC = 'A valid CWS measure exists for this measurement point'

#: str: Flag name for ascent data
FLG_PRELAUNCH = 'is_prelaunch_data'
#: str: Flag desc for ascent data
FLG_PRELAUNCH_DESC = 'Prelaunch data'

#: str: Flag name for ascent data
FLG_ASCENT = 'is_ascent_data'
#: str: Flag desc for ascent data
FLG_ASCENT_DESC = 'Ascent data'

#: str: Flag name for descent data
FLG_DESCENT = 'is_descent_data'
#: str: Flag desc for descent data
FLG_DESCENT_DESC = 'Descent data'

#: str: Flag name for PBL points
FLG_PBL = 'is_in_PBL'
#: str: Flag desc for PBL points
FLG_PBL_DESC = 'Measurement is in the PBL (i.e. x < PBLH)'

#: str: Flag name for free troposphere points
FLG_FT = 'is_in_FT'
#: str: Flag desc for free troposphere points
FLG_FT_DESC = 'Measurement is in the FT (i.e. x < tropopause and x > PBLH)'

#: str: Flag name for UTLS points
FLG_UTLS = 'is_in_UTLS'
#: str: Flag desc for UTLS points
FLG_UTLS_DESC = 'Measurement is in the UTLS'

#: str: Flag name for MUS points
FLG_MUS = 'is_in_MUS'
#: str: Flag desc for MUS points
FLG_MUS_DESC = 'Measurement is in the MUS'


# Metadata special fields
# -----------------------

#: str: Metdata field to store the GNSS time of the first point in the profile.
MTDTA_FIRST = 'first_timestamp'

#: str: Metadata field to store the GNSS time of the manufacturer-detected launch.
MTDTA_LAUNCH = 'launch_timestamp'

#: str: Metadata field to store the GNSS time of the manufacturer-detected balloon burst.
MTDTA_BURST = 'burst_timestamp'

#: str: Metadata field to store the tropopause geopotential height
MTDTA_TROPOPAUSE = 'dvas_tropopause'

#: str: Metadata field to store the tropopause geopotential height
MTDTA_PBLH = 'dvas_PBLH'

#: str: Metadata field to store the min geopotential height of the UTLS region
MTDTA_UTLSMIN = 'dvas_UTLS_min'

#: str: Metadata field to store the max geopotential height of the UTLS region
MTDTA_UTLSMAX = 'dvas_UTLS_max'

#: str: Metadata field to store the max geopotential height of the UTLS region
MTDTA_MUSMIN = 'dvas_MUS_min'

#: str: Metadata field to store the synop cloud code
MTDTA_SYNOP = 'synop_clouds'
