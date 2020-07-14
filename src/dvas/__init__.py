"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

from pathlib import Path

from .dvas_version import VERSION

__version__ = VERSION

pkg_path = Path(__file__).absolute().parent
expl_path = pkg_path / '..' / '..' / 'examples'
