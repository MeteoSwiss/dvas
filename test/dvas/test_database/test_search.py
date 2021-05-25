"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.database.search module.

"""

# Import external packages and modules
import numpy as np

# Import from python packages and modules under test
from dvas.database.search import SearchInfoExpr
from dvas.hardcoded import TAG_RAW_NAME, TAG_GDP_NAME

# Define db_data
db_data = {
    'sub_dir': 'test_search',
    'data': [
        {
            'index': np.array([0, 1, 2]),
            'value': np.array([500, 501, 502]),
            'prm_name': 'temp',
            'info': {
                'edt': '20200101T0000Z',
                'mdl_name': 'YT',
                'srn': 'YT-100', 'pid': '0',
                'tags': ('data_test_db', 'e:1', 'r:1', arg_tag),
                'metadata': {'test_key_str': 'one', 'test_key_num': '1'},
                'src': ''
            },
        } for arg_tag in [TAG_RAW_NAME, TAG_GDP_NAME]
    ]
}


def test_search_event_expr_eval(db_init):
    """Test SearchInfoExpr.eval static function"""

    # Define
    data = db_init.data
    args = (data[0]['prm_name'], True)
    SearchInfoExpr.set_stgy('info')

    # Test all
    assert len(SearchInfoExpr.eval('all()', *args)) > 0

    # Test datetime
    assert (
        SearchInfoExpr.eval(
            'datetime("20200101T0000Z", "==")', * args
        ) ==
        SearchInfoExpr.eval(
            'datetime("20200101T0000Z")', *args
        ) ==
        SearchInfoExpr.eval(
            'dt("2020-01-01 00:00:00+00:00")', *args
        ) !=
        SearchInfoExpr.eval(
            'datetime("20180110T0000Z", "==")', *args
        )
    )

    # Test not_ and or_
    assert (
        SearchInfoExpr.eval(
            'datetime("20200101T0000Z", "==")', *args
        ) ==
        SearchInfoExpr.eval(
            'not_(or_(datetime("20200101T0000Z", "<"), datetime("20200101T0000Z", ">")))',
            *args
        ) !=
        SearchInfoExpr.eval(
            'datetime("20180110T0000Z", "==")', *args
        )
    )

    # Test tag
    assert len(
        SearchInfoExpr.eval(
            f'tags(("e:1", "r:1"))', *args
        )
    ) > 0
    assert (
        SearchInfoExpr.eval(
            'tags(("e:1", "r:1"))', *args
        ) ==
        SearchInfoExpr.eval(
            'or_(tags("e:1"), tags("r:1"))',
            *args
        )
    )

    # Test serial number
    assert len(
        SearchInfoExpr.eval(
            f'srn("{data[0]["info"]["srn"]}")', *args
        )
    ) > 0

    # Test and_
    assert (
        SearchInfoExpr.eval(
            f'and_(tags("{data[0]["info"]["tags"][1]}"), not_(tags("{data[0]["info"]["tags"][1]}")))', *args
        ) ==
        SearchInfoExpr.eval(
            'not_(all())',
            *args
        )
    )

    # Test raw()
    assert SearchInfoExpr.eval(f"tags('raw')", *args) == SearchInfoExpr.eval(f"raw()", *args)

    # Test gdp()
    assert SearchInfoExpr.eval(f"tags('gdp')", *args) == SearchInfoExpr.eval(f"gdp()", *args)
