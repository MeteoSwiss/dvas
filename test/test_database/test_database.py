"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.database.database module.

"""

# Import from python packages and modules
from dvas.database.database import SearchEventExpr


def test_search_event_expr_eval():
    """Test SearchEventExpr.eval static function"""

    # Define
    args = ('trepros1', True)

    # Test all
    assert len(SearchEventExpr.eval('all()', *args)) > 0

    # Test datetime
    assert (
        SearchEventExpr.eval(
            'datetime("20180110T0000Z", "==")', *args
        ) ==
        SearchEventExpr.eval(
            'datetime("20180110T0000Z")', *args
        ) ==
        SearchEventExpr.eval(
            'dt("20180110T0000Z")', *args
        )
    )

    # Test not_ and or_
    assert (
        SearchEventExpr.eval(
            'datetime("20180110T0000Z", "==")', *args
        ) ==
        SearchEventExpr.eval(
            'not_(or_(datetime("20180110T0000Z", "<"), datetime("20180110T0000Z", ">")))',
            *args
        )
    )

    # Test tag
    assert (
        SearchEventExpr.eval(
            'tag(("e1", "r1"))', *args
        ) ==
        SearchEventExpr.eval(
            'or_(tag("e1"), tag("r1"))',
            *args
        )
    )

    # Test and_
    assert (
        SearchEventExpr.eval(
            'and_(tag("e1"), not_(tag("e1")))', *args
        ) ==
        SearchEventExpr.eval(
            'not_(all())',
            *args
        )
    )
