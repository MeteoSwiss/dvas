"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: examples
"""

from pathlib import Path

# Set the data path to look where we currently are
from dvas.dvas_environ import path_var
# WARNING: this must be done BEFORE importing anything else ... !
path_var.orig_data_path = Path(__file__).parent / 'data'
path_var.config_dir_path = Path(__file__).parent / 'config'

# Import
from dvas.data.data import MultiProfile, MultiRSProfile, MultiGDPProfile
from dvas.data.data import update_db
from dvas.dvas_logger import LogManager
from dvas.database.database import DatabaseManager

from dvas.dvas_logger import dvasError

from dvas.data.linker import LocalDBLinker
from dvas.database.model import Data
from dvas.data.strategy.load import LoadProfileStrategy


if __name__ == '__main__':

    # Define
    RESET_DB = False

    filt_gdp = "tag('gdp')"
    filt_gdp_der = "and_(tag('derived'), tag('gdp'))"
    filt_raw = "and_(tag('raw'), not_(tag('gdp')))"
    filt_all = "all()"
    filt_der = "and_(tag('derived'), not_(tag('gdp')))"

    # Create database
    db_mngr = DatabaseManager(reset_db=RESET_DB)

    # Update DB + log
    if RESET_DB:
        with LogManager():
            update_db('tdtpros1', strict=True)
            update_db('trepros1', strict=True)
            update_db('treprosu_r', strict=True)
            update_db('treprosu_s', strict=True)
            update_db('treprosu_t', strict=True)
            update_db('altpros1', strict=True)



    # # Load a basic profile, with a variable, and altitude.
    # prf = MultiProfile()
    # try:
    #     prf.load(filt_der, 'trepros1', 'altpros1')
    #
    # except dvasError:
    #     prf.load(filt_raw, 'trepros1', 'altpros1')
    #     prf.sort()
    #     prf.save()

    # Load a basic time profile, with a variable and altitude
    rs_prf = MultiRSProfile()
    try:
        rs_prf.load(filt_der, 'trepros1', 'tdtpros1', alt_abbr='altpros1')

    except dvasError:
        rs_prf.load(filt_raw, 'trepros1', 'tdtpros1', alt_abbr='altpros1')
        rs_prf.sort()
        #     rs_prf.resample()
        rs_prf.save()

    print(rs_prf.get_info('evt_id'))
    print(rs_prf.get_info('rig_id'))
    print(rs_prf.get_info('gdp_mdl_id'))


    #
    # # Load the GDPs for temperature, including all the errors
    # gdp_prf = MultiGDPProfile()
    # try:
    #     gdp_prf.load(
    #         filt_gdp_der, 'trepros1', 'tdtpros1', alt_abbr='altpros1',
    #         ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t'
    #     )
    #
    # except dvasError:
    #     gdp_prf.load(
    #         filt_gdp, 'trepros1', 'tdtpros1', alt_abbr='altpros1',
    #         ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t'
    #     )
    #     gdp_prf.sort()
    #     gdp_prf.resample()
    #     gdp_prf.save()

    #
    # # Make a plot
    # prf.plot(fig_num=1, save_fn='plot1')
    #
    # # Load a regular radiosonde profile, with a variable, altitude, and time deltas.
    # rs_prf = MultiRSProfile()
    # rs_prf.load(filt_in, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1', inplace=True)
    #
    # # Load the GDPs for temperature, including all the errors
    # gdp_prf = MultiGDPProfile()
    # gdp_prf.load(filt_in, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1',
    #              ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t', inplace=True)
    #
    # # Make a plot
    # gdp_prf.plot(fig_num=2, x='tdt', save_fn='plot2')
    #
    # # Use convenience getters to extract some info
    # srns = gdp_prf.get_evt_prm('sn')
    # vals_alt = gdp_prf.get_prms(['val', 'alt'])
    #
    # # Compute the total error from GDPs
    # uc_tot = gdp_prf.uc_tot
    #
    # # Set some data to 0 just to check
    # rs_prf.profiles['rs_prf'][0].data['val'] = 0
    #
    # # Save this back to the DB with new tags
    # rs_prf.save(add_tags=['vof'], rm_tags=['raw'], prm_list=['val', 'alt', 'tdt'])
    #
    # # Extract it, and check it is still modified
    # rs_prf_vof = MultiRSProfile()
    # rs_prf_vof.load(filt_vof, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1', inplace=True)
    #


""" Original code ... kept for now until everything has been reconnected.
    # Define
    RESET_DB = False
    filt = "tag('e1')"

    # Create database
    db_mngr = DatabaseManager(reset_db=RESET_DB)

    # Update DB + log
    if RESET_DB:
        with LogManager():
            update_db('trepros1', strict=True)
            update_db('treprosu_')
            update_db('altpros1')
            update_db('prepros1')

    # Time
    data_t = time_mngr.load(filt, 'trepros1')
    data_s = data_t.sort()
    data_r = data_s.resample()
    data_sy = data_r.synchronize()
    data_sy.plot()
    data_sy.save({'data': 'dummy_3'})
    test = time_mngr.load(filt, 'dummy_3')
    test = test.sort()
    data_sy.plot()

    # Alt
    data_t = alt_mngr.load(filt, 'trepros1', 'altpros1')
    data_s = data_t.sort()
    data_r = data_s.resample()
    data_sy_t = data_r.synchronize(method='time')
    data_sy_a = data_r.synchronize(method='alt')
    data_sy_t.save({'data': 'dummy_0', 'alt': 'dummy_1'})
    test = data_t = alt_mngr.load(filt, 'dummy_0', 'dummy_1')
    test = test.sort()
    data_sy_t.plot()
"""
