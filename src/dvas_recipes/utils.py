"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains general dvas_recipe utilities.
"""

from pathlib import Path

# Import from this module
from . import dynamic


def demo_storage_path():
    """ Returns the absolute path to the stored dvas demo locations, from where the dvas demo
    scripts can be copied locally.

    """
    return Path(__file__).resolve(strict=True).parent / '..' / 'dvas_demo'


def configs_storage_path():
    """ Returns the absolute path to the stored dvas configs locations, from where the dvas default
    config files can be copied locally.

    """
    return Path(__file__).resolve(strict=True).parent / 'configs'


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


def fn_suffix(fid=None, eid=None, rid=None, var=None, mids=None, pids=None, tags=None):
    """ Returns the default suffix of filenames given a set of info provided by the user.

    Args:
        fid (str, optional): the flight id
        eid (int, optional): the event id
        rid (int, optional): the rig id
        var (str, optional): the variable name
        mids (list of str, optional): the list of mids
        pids (list of str, optional): the list of pids
        tags (list of str, optional): the list of tags associated with the data

    Returns:
        str: the filename suffix, that can be fed to the `dvas.plots.utils.fancy_savefig()`.
    """

    suffix = ''
    for item in [fid, eid, rid, var]:
        if item is not None:
            suffix += '_{}'.format(item.replace(':', ''))

    if mids is not None:
        suffix += '_{}'.format('-'.join(mids))

    if pids is not None:
        suffix += '_{}'.format('-'.join(pids))

    if tags is not None:
        suffix += '_{}'.format('-'.join([item.replace('tod:', '').replace(':', '')
                               for item in tags]))

    return suffix[1:] if len(suffix) > 0 else None


def format_tags(tags):
    """ Formats a list of tags according to some basic rules. Any tag set by the user should be
    fed to this routine.

    Args:
        tags (str, list): the tags to format

    Returns:
        list: the cleaned-up tags
    """

    # First, be nice with str inputs = single tag provided.
    if isinstance(tags, str):
        tags = [tags]

    return tags


def rsid_tags(pop=None):
    """ Returns the list of rsid (recipe step ID) tags, possibly by removing some of them.

    Args:
        pop (list, optional): if set, any tags in this list will be removed from the returned list.

    Returns:
        list: rsid tags, minus the popped ones.
    """

    tags_out = dynamic.ALL_STEP_IDS

    # If warranted, remove some of the tags ...
    if pop is not None and tags_out is not None:
        if isinstance(pop, str):
            pop = [pop]
        tags_out = [item for item in tags_out if item not in pop]

    return tags_out


def cws_vars(incl_latlon=False):
    """ Return the list of variables that are present in CWS. """

    noncws_vars = ['wvec']

    if not incl_latlon:
        noncws_vars += ['lat', 'lon']

    return {var_name: var_content for (var_name, var_content) in dynamic.ALL_VARS.items()
            if var_name not in noncws_vars}
