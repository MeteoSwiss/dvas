"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: high-level GDP recipes for the UAII22 campaign
"""


def build_cws(evt_id, rig_id, **kwargs):
    """ Highest-level function responsible for assembling the cokbined working standards for a
    specific RS flight.

    This function directly builds the profiles and upload them to the db with the 'cws' tag.

    Args:
        evt_id (str|int): event id to be synchronized, e.g. 80611
        rig_id (str|int): rig id to be synchronized, e.g. 1
        **kwargs: keyword arguments to be fed to the underlying cws-building routines.

    """
