"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: UAII2022 recipe tools
"""


def get_query_filter(tags_in: list = None, tags_out: list = None, mids: list = None) -> str:
    """ Assembles a str to query the dvas DB, given a list of tags to include and/or exclude.

    Args:
        tags_in (list, optional): list of tags required to be present
        tags_out (list, optional): list of tags required to be absent
        mids (list, optional): list of mids required

    Returns:
        str: the query filter
    """

    filt = []

    if tags_in is not None:
        filt += ["tags('" + "'), tags('".join(tags_in) + "')"]

    if tags_out is not None:
        filt += ["not_(tags('" + "')), not_(tags('".join(tags_out) + "'))"]

    if mids is not None:
        filt += ["mid('" + "'), mid('".join(mids) + "')"]

    if len(filt) == 0:
        return ''

    return 'and_(' + ', '.join(filt) + ')'
