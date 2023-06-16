"""
Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: export recipes for the UAII2022 campaign
"""

# Import general Python packages
import logging
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from netCDF4 import Dataset  # noqa pylint: disable=E0611

# Import dvas modules and classes
from dvas.logger import log_func_call
from dvas.dvas import Database as DB
from dvas.environ import path_var as dvas_path_var
from dvas.version import VERSION
from dvas.data.data import MultiRSProfile, MultiGDPProfile, MultiCWSProfile
from dvas.hardcoded import PRF_TDT, PRF_ALT, PRF_VAL, PRF_FLG, PRF_IDX
from dvas.hardcoded import TAG_ORIGINAL, TAG_CLN, TAG_1S, TAG_SYNC, TAG_GDP, TAG_CWS


# Import from dvas_recipes
from .. import dynamic
from ..recipe import for_each_flight
from ..errors import DvasRecipesError
from .. import utils as dru
from . import tools

logger = logging.getLogger(__name__)


def get_nc_fname(suffix: str, fid: str, edt: str, typ: str, mid: str = None, pid: str = None):
    """ Assemble the netCDF filename for a given flight/mid.

    Args:
        suffix (str): filename suffix.
        fid (str): Flight id.
        edt (str): formatted event datetime, e.g. 'YYYYMMDDThhmmss'.
        typ (str): 'cws', 'gdp', or 'mdp'.
        mid (str, optional): model id. Useless if typ == 'cws'. Defaults to None.
        pid (str, optional): product id. Useless if typ == 'cws'. Defaults to None.

    Returns:
        str: the netcdf filename

    """

    # Let's create the netCDF file.
    fname = f"{suffix}_{fid}_{edt}_"
    if typ == 'cws':
        return fname + "CWS.nc"
    else:
        if typ == 'gdp':
            fname += 'GDP_'
        elif typ == 'mdp':
            fname += 'MDP_'
        else:
            raise DvasRecipesError(f' typ unknown: {typ}')

        return fname + f"{mid}_{pid}.nc"


def add_cf_attributes(grp, title='', institution: str = '', comment: str = None):
    """" Add global UAII attributes to a netCDF group.

    Args:
        grp: netCDF Dataset or group to add the info to.

    """

    setattr(grp, 'Conventions', "CF-1.7")
    setattr(grp, 'title', title)
    # History
    # Thanks to Craig McQueen's reply for getting timezone-aware utcnow() time on
    # https://stackoverflow.com/questions/2331592
    val = f'{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f %Z")} - '
    val += f'created with dvas {VERSION}'
    setattr(grp, 'history', val)
    setattr(grp, 'institution', institution)
    # Source
    val = "/".join([item.name for item in Path(__file__).parents[:3][::-1]])
    val += f"/{Path(__file__).name}"
    val = f'dvas recipe: {dynamic.RECIPE} (step id: {dynamic.CURRENT_STEP_ID}) [{val}]'
    setattr(grp, 'source', val)

    # Comment
    if comment is not None:
        setattr(grp, 'comment', comment)

    # References
    setattr(grp, 'references', 'See the UAII 2022 Final Report for details.')


def set_attribute(grp, name, value):
    """ Safe attribute setting function, with in-built check mecanism in case the attribute already
    exists.

    Args:
        grp: netCDF Dataset or group
        name (str): attribute name
        value: attribute value

    """

    try:
        # If the Attribute already exists, check that the values match
        stored_value = grp.getncattr(name)
        assert stored_value == value, f'{name} attribute value mismatch: {stored_value} vs {value}'
    except AttributeError:
        # Else, set the attribute
        setattr(grp, name, value)


def add_dvas_attributes(grp, prf):
    """ Add dvas-related Global Attributes to a netCDF file.

    Args:
        grp: netCDF Dataset or group.
        prf: MultiProfile, MultiRSProfile, MultiGDPProfile, MultiCWSProfile of length 1.

    """

    if len(prf) != 1:
        raise DvasRecipesError(f'I got {len(prf)} profiles instead of 1.')

    # Extract the oid, which I need to identify important profile info
    oid = prf[0].info.oid

    # Let's also get a view of the db to fetch the info via the oid
    db_view = DB.extract_global_view()
    # Down select only the relevant rows
    db_view = db_view[db_view.oid.isin(oid)]

    # Start setting things
    set_attribute(grp, 'd.ObjectId', ','.join([str(item) for item in oid]))

    # Add dvas specific global attributes
    (fid, eid, rid) = dynamic.CURRENT_FLIGHT
    set_attribute(grp, 'd.Flight.Id', f"{fid}")
    assert len(db_view.eid.unique()) == 1, 'eid mismatch'
    set_attribute(grp, 'd.Flight.EventId', f"{eid}")
    assert len(db_view.edt.unique()) == 1, 'edt mismatch'
    set_attribute(grp, 'd.Flight.Datetime',
                  f"{prf[0].info.edt.strftime('%Y-%m-%dT%H:%M:%S.%f %Z')}")
    set_attribute(grp, 'd.Flight.RigId', f"{rid}")
    tods = []
    for item in ['daytime', 'nighttime', 'twilight']:
        if prf[0].has_tag(f'tod:{item}'):
            tods += [item]
    set_attribute(grp, 'd.Flight.TimeOfDay', ','.join(tods))

    set_attribute(grp, 'd.Sonde.SerialNumber', f"{','.join(db_view.srn)}")
    set_attribute(grp, 'd.Sonde.ModelId', f"{','.join(db_view.mid)}")
    set_attribute(grp, 'd.Sonde.ModelName', f"{','.join(db_view.mdl_name)}")
    set_attribute(grp, 'd.Sonde.ModelDescription', f"{','.join(db_view.mdl_desc)}")
    set_attribute(grp, 'd.GroundSystem.Id', f"{','.join(db_view.pid)}")

    set_attribute(grp, 'd.Data.Source', f"{prf[0].info.src}")
    set_attribute(grp, 'd.Data.IsOriginal', f"{prf[0].has_tag(TAG_ORIGINAL)}")
    set_attribute(grp, 'd.Data.IsCleaned', f"{prf[0].has_tag(TAG_CLN)}")
    set_attribute(grp, 'd.Data.IsResampled', f"{prf[0].has_tag(TAG_1S)}")
    set_attribute(grp, 'd.Data.IsSynchronized', f"{prf[0].has_tag(TAG_SYNC)}")

    # Also add all the metadata present
    for (key, value) in prf[0].info.metadata.items():
        # Skip the fid, that was already stored more cleanly earlier
        if key == 'fid':
            continue
        # Format datetimes cleanly
        if isinstance(value, datetime):
            value = value.strftime('%Y-%m-%dT%H:%M:%S.%f %Z')
        set_attribute(grp, f'd.Metadata.{key}', f"{value}")


def add_nc_variable(grp, prf):
    """ Fills a netCDF Dataset with the Profile data.

    Args:
        grp: netCDF Dataset or group
        prf: MultiProfile, MultiRSProfile, MultiGDPProfile, MultiCWSProfile of length 1.

    """

    if len(prf) != 1:
        raise DvasRecipesError(f'I got {len(prf)} profiles instead of 1.')

    # Create the netCDF base dimensions, if it does not already exists
    # I shall create a single dimension based on the step id (idx). The time is that of
    # the radiosonde, and essentially useless at this stage (no sync possible without a
    # lot more info than available). So is the reference altitude axis, which is
    # just a copy of the gph variable.
    if len(grp.dimensions.keys()) == 0:
        grp.createDimension("relative_time", len(prf[0].data))
        rel_time = grp.createVariable("relative_time", "i8", dimensions=("relative_time"))
        rel_time[:] = prf[0].data.index.get_level_values(PRF_IDX).values
        rel_time.units = 's'
        rel_time.standard_name = 'relative_time'
        rel_time.long_name = 'Relative time'
        rel_time.comment = 'Seconds counted since d.Metadata.first_timestamp'
        # rel_time.axis = 'Relative time'

    else:
        # Check the dimension is actually still matching !
        if (a := grp.dimensions['relative_time'].size) != (b := len(prf[0].data)):
            raise DvasRecipesError(
                f'Size mismatch: relative_time dimension ({a}) vs prf[0] ({b}) ')

    # Now store all the columns, not forgetting the total uncertainty
    for col in prf[0].data.columns.tolist() + ['uc_tot']:
        # Flags should be stored as int, everything else as float
        # For the flags, we also must take care of NaNs.
        if col in [PRF_FLG]:
            nc_tpe = 'i8'
            np_type = 'int64'
            na_value = 0
        else:
            nc_tpe = 'f8'
            np_type = 'float64'
            na_value = np.nan

        # If the  variable was not loaded (i.e. no data ever loaded for it), move on.
        if (col != 'uc_tot') and (col not in prf.var_info.keys()):
            continue
        # Idem for uc_tot - not present in RSProfiles
        if (col == 'uc_tot') and (not hasattr(prf[0], 'uc_tot')):
            continue

        # Create the actual nc Variable
        var_name = prf.var_info[PRF_VAL]["prm_name"]
        if col not in [PRF_VAL]:
            var_name += f"_{col}"
        var_nc = grp.createVariable(f'{var_name}', nc_tpe, dimensions=("relative_time"))

        # Fill the data
        var_nc[:] = getattr(prf[0], col).to_numpy(dtype=np_type, na_value=na_value)

        # Set the Variable attributes
        if col != 'uc_tot':
            setattr(var_nc, 'long_name', prf.var_info[col]['prm_desc'])
            setattr(var_nc, 'units', prf.var_info[col]['prm_unit'])
            setattr(var_nc, 'comment', prf.var_info[col]['prm_cmt'])
        else:
            # TODO: here the k-level is hardcoded !!! This is very dangerous !
            setattr(var_nc, 'long_name',
                    f"{prf.var_info[PRF_VAL]['prm_desc']} total uncertainty (k=1)")
            setattr(var_nc, 'units', prf.var_info[PRF_VAL]['prm_unit'])
            setattr(var_nc, 'comment', 'uc_tot = sqrt(ucs**2 + uct**2 + ucu**2)')

        # Specify the flag codes
        if col == PRF_FLG:
            masks = np.array([2**item[1]['bit_pos'] for item in prf[0].flg_names.items()])
            meanings = np.array([item[1]['flg_name'] for item in prf[0].flg_names.items()])
            sort_order = np.argsort(masks)
            setattr(var_nc, 'flag_masks', f"{', '.join([str(item) for item in masks[sort_order]])}")
            setattr(var_nc, 'flag_meaning', f"{', '.join(meanings[sort_order])}")


@for_each_flight
@log_func_call(logger, time_it=False)
def export_profiles(tags: str | list, which: str | list, suffix: str = '', institution: str = ''):
    """ Export profiles from the db to netCDF

    Args:
        tags (str|list): list of tags to identify profiles to export in the DB.
        which (str|list): list of profiles to include, e.g.: ['mdp', 'gdp', 'cws'].
        suffix (str, optional): name of the netCDF file suffix. Defaults to ''.
        institution (str, optional): for the netCDF eponym field. Defaults to ''.

    """

    # Format the search tags
    tags = dru.format_tags(tags)

    # Extract the flight info
    (fid, eid, rid) = dynamic.CURRENT_FLIGHT
    logger.info('Exporting %s profiles for (%s; %s; %s) ...', which, fid, eid, rid)

    # What is the destination for the nc files ?
    out_path = dvas_path_var.output_path
    if out_path is None:
        raise DvasRecipesError('Output path is None.')
    if not out_path.exists():
        # Be bold and create the folder, if it does not yet exist
        logger.info('Creating output path {out_path} ...')
        out_path.mkdir(parents=True)
        # Set user read/write permission
        out_path.chmod(out_path.stat().st_mode | 0o600)

    # Let's figure out which MDPs and GDPs exist for this fid/eid/rid
    db_view = DB.extract_global_view()
    mids = db_view.loc[(db_view.eid == eid) * (db_view.rid == rid)]
    # Let's also extract the edt, which I will need later on ...
    edt = np.unique(mids['edt'])
    if len(edt) != 1:
        raise DvasRecipesError(f'There should be only 1 edt per eid/rid pair, not: {len(edt)}')
    else:
        edt = edt[0]

    comment = "Cleaned-up, resampled, synced atmospheric profile."

    # Let's deal with the CWS first, if warranted
    if 'cws' in which:
        logger.info('Exporting the cws ...')

        # Let's define the full name and create the netCDF file
        fname = get_nc_fname(suffix, fid, edt.strftime('%Y%m%dT%H%M%S'), 'cws')
        rootgrp = Dataset(out_path / fname, "w", format="NETCDF4")

        # Add the Global Attributes
        add_cf_attributes(rootgrp, title='Combined Working measurement Standard (CWS)',
                          institution=institution, comment=comment)

        # Assemble the search filter
        filt = tools.get_query_filter(tags_in=tags + [eid, rid, TAG_CWS],
                                      tags_out=None,
                                      )

        # Let's keep track of the CWS length, and make sure all the profiles have the same length
        len_cws = None

        # Start looking for all the variables
        for (var_name, var) in dru.cws_vars(incl_latlon=True).items():

            prf = MultiCWSProfile()
            prf.load_from_db(f'{filt}', var_name,
                             tdt_abbr=dynamic.INDEXES[PRF_TDT],
                             alt_abbr=dynamic.INDEXES[PRF_ALT],
                             ucs_abbr=var['ucs'],
                             uct_abbr=var['uct'],
                             ucu_abbr=var['ucu'],
                             inplace=True)

            add_dvas_attributes(rootgrp, prf)
            add_nc_variable(rootgrp, prf)

            if len_cws is None:
                len_cws = len(prf[0].data)
            else:
                assert len_cws == (b := len(prf[0].data)), \
                    f'CWS {var_name} length is {b}. Should be {len_cws}'

    # Now let's deal with MDPs and GDPs
    # Let's iterate over all the mids of this flight
    for (_, item) in mids.iterrows():

        # Process specific mdps/gpds only if warranted
        if item['is_gdp'] and 'gdp' not in which:
            continue
        if not item['is_gdp'] and 'mdp' not in which:
            continue

        logger.info('Exporting %s [#%s] ...', item['mid'], item['pid'])

        # Let's create the netCDF file.

        if item['is_gdp']:
            typ = 'gdp'
            title = 'GRUAN Data Product (GDP) atmospheric profile'
        else:
            typ = 'mdp'
            title = 'Manufacturer Data Product (MDP) atmospheric profile'
        fname = get_nc_fname(suffix, fid, edt.strftime('%Y%m%dT%H%M%S'), typ,
                             mid=item['mid'], pid=item['pid'])
        rootgrp = Dataset(out_path / fname, "w", format="NETCDF4")

        # Add CF-1.7 Global Attributes
        add_cf_attributes(rootgrp, title=title, institution=institution, comment=comment)

        # Now get the data from the DB
        for (var_name, var) in dru.cws_vars(incl_latlon=typ == 'gdp').items():

            # Assemble the search filter
            filt = tools.get_query_filter(tags_in=tags + [eid, rid],
                                          tags_out=[TAG_CWS],
                                          oids=[item['oid']])

            if item['is_gdp']:
                prf = MultiGDPProfile()
                prf.load_from_db(f'and_({filt}, tags("{TAG_GDP}"))', var_name,
                                 tdt_abbr=dynamic.INDEXES[PRF_TDT],
                                 alt_abbr=dynamic.INDEXES[PRF_ALT],
                                 ucs_abbr=var['ucs'],
                                 uct_abbr=var['uct'],
                                 ucu_abbr=var['ucu'],
                                 inplace=True)
            else:
                prf = MultiRSProfile()
                prf.load_from_db(f'and_({filt}, not_(tags("{TAG_GDP}")))', var_name,
                                 tdt_abbr=dynamic.INDEXES[PRF_TDT],
                                 alt_abbr=dynamic.INDEXES[PRF_ALT],
                                 inplace=True)

            add_dvas_attributes(rootgrp, prf)
            add_nc_variable(rootgrp, prf)

            if len_cws is None:
                len_cws = len(prf[0].data)
            else:
                assert len_cws == (b := len(prf[0].data)), \
                    f'{item["mid"]} [#{item["pid"]}] {var_name} length is {b}. Should be {len_cws}'

        rootgrp.close()
