"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains general dvas_recipe utilities.
"""

from pathlib import Path


def demo_storage_path():
    """ Returns the absolute path to the stored dvas demo locations, from where the dvas demo
    scripts can be copied locally.

    """
    return Path(__file__).resolve(strict=True).parent / '..' / 'dvas_demo'


def recipe_storage_path():
    """ Returns the absolute path to the stored dvas recipes location, from where they can be
    copied locally.

    """
    return Path(__file__).resolve(strict=True).parent / 'recipes'


def default_arena_path():
    """ Returns the default **relative** location and name of the dvas processing arena for a
    specific recipe: './dvas_recipe_arena/'

    Args:

    Returns:
        pathlib.Path: the default relative Path.
    """

    return Path('.', 'dvas_proc_arena')


def fn_suffix(eid=None, rid=None, var=None, tags=None):
    """ Returns the default suffix of filenames given a set of info provided by the user.

    Args:
        eid (int, optional): the event id
        rid (int, optional): the rig id
        var (str, optional): the variable name
        tags (list of str, optional): the list of tags associated with the data

    Returns:
        str: the filename suffix, that can be fed to the `dvas.plots.utils.fancy_savefig()`.
    """

    suffix = ''
    for item in [eid, rid, var]:
        if item is not None:
            suffix += '_{}'.format(item)

    if tags is not None:
        suffix += '_{}'.format('-'.join(tags))

    return suffix[1:] if len(suffix) > 0 else None
