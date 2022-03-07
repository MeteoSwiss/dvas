"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: UAII2022 recipe tools
"""


def get_query_filter(tags_in: list = None, tags_out: list = None) -> str:
    """ Assembles a str to query the dvas DB, given a list of tags to include and/or exclude.

    Args:
        tags_in (list, optional): list of tags required to be present
        tags_out (list, optional): list of tags required to be absent

    Returns:
        str: the query filter
    """

    if tags_in is not None:
        filt_in = "tags('" + "'), tags('".join(tags_in) + "')"
    else:
        filt_in = ''

    if tags_out is not None:
        filt_out = "not_(tags('" + "')), not_(tags('".join(tags_out) + "'))"
    else:
        filt_out = ''

    return f'and_({filt_in}, {filt_out})'
