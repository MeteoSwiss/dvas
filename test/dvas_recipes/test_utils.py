"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas_recipes.utils module.

"""


from dvas_recipes import utils as dru


def test_fn_suffix():
    """ A test function to make sure the automated suffix for the dvas_recipes work as intended."""

    eid = '12345'
    rid = '1'
    var = 'var'
    tags = ['tag1', 'tag2']

    assert dru.fn_suffix(eid=eid, rid=rid, var=var, tags=tags) == '12345_1_var_tag1-tag2'
    assert dru.fn_suffix() is None
