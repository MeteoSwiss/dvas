"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: hardcoded parameters and variables for dvas.

"""

from pathlib import Path

#: pathlib.Path: Package path
PKG_PATH = Path(__file__).resolve(strict=True).parent

#: pathlib.Path: Matplotlib plots styles
MPL_STYLES_PATH = Path('plots/mpl_styles')

#: int: Config generator limit
CONFIG_GEN_LIM = 10000

#: list of str: Config file extension
CONFIG_FILE_EXT = ['yml', 'yaml']

#: str: Event id regexp pattern
EVT_ID_PAT = r'^e:\w+$'

#: str: Rig id regexp pattern
RIG_ID_PAT = r'^r:\w+$'

#: str: Product id regexp pattern
PRD_ID_PAT = r'^p:\w+$'

#: str: Model id regexp pattern
MDL_ID_PAT = r'^m:\w+$'

#: str: Name of the integer index for the pandas DataFrame of Profiles, RSProfile, GDPProfiles
PRF_REF_INDEX_NAME = '_idx'

