"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: hardcoded parameters and variables for dvas.

"""

from pathlib import Path

#: pathlib.Path: package path
PKG_PATH = Path(__file__).resolve(strict=True).parent

#: pathlib.Path: processing arena path
PROC_PATH = Path('.')

#: str: Name of the integer index for the pandas DataFrame of Profiles, RSProfile, GDPProfiles
PRF_REF_INDEX_NAME = '_idx'
