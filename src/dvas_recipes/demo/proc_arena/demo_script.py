"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: demo code that illustres the core dvas functionalities
"""

# Import python package
from pathlib import Path

# Import stuff from dvas
import dvas.plots.utils as dpu
from dvas.dvas import start_log
from dvas.data.data import MultiProfile
from dvas.data.io import update_db
from dvas.database.database import DatabaseManager
from dvas.environ import path_var

# Define
demo_file_path = Path(__file__).resolve()

if __name__ == '__main__':

    # Init path
    path_var.config_dir_path = demo_file_path.parent / 'config'
    path_var.orig_data_path = demo_file_path.parent / 'data'
    path_var.local_db_path = demo_file_path.parent / 'db'
    path_var.output_path = demo_file_path.parent / 'output'

    # Start the logging
    start_log(3, level='DEBUG')  # 3 = log to screen only.

    # Fine-tune the plotting behavior of dvas
    dpu.set_mplstyle('nolatex') # The safe option. Use 'latex' fo prettier plots.

    # The generic formats to save the plots in
    dpu.PLOT_FMTS = ['png', 'pdf']

    # Show the plots on-screen ?
    dpu.PLOT_SHOW = False

    # Reset the DB to "start fresh" ?
    RESET_DB = True

    # Define some basic search queries
    filt_gdp = "tags('gdp')"
    filt_raw = "tags('raw')"
    filt_raw_not = "not_(tags('raw'))"
    filt_all = "all()"
    filt_dt = "dt('20171024T120000Z', '==')"

    # Define some more complex queries
    filt_gdp_dt = "and_(%s, %s)" % (filt_gdp, filt_dt)

    # Create the dvas database
    db_mngr = DatabaseManager(reset_db=RESET_DB)

    # Update the database
    if RESET_DB:
        update_db('tdtpros1', strict=True)
        update_db('trepros1', strict=True)
        update_db('trepros1_flag', strict=True)
        update_db('treprosu_r', strict=True)
        update_db('treprosu_s', strict=True)
        update_db('treprosu_t', strict=True)
        update_db('altpros1', strict=True)

    # Load a basic profile, with a variable, and altitude.
    prf = MultiProfile()
    prf.load_from_db(filt_raw, 'trepros1', 'altpros1')

    prf.save_to_db(add_tags='vof')

    prf1 = MultiProfile()
    prf1.load_from_db("tags('vof')", 'trepros1', 'altpros1')

    prf1.profiles[0].set_flg('raw_na', True)


    # # Load a basic time profile, with a variable and altitude
    # rs_prf = MultiRSProfile()
    # rs_prf.load_from_db(filt_dt, 'trepros1', 'tdtpros1', alt_abbr='altpros1')
    # rs_prf.sort()
    # rs_prf.save_to_db()
    #
    #
    #
    # # Acccess some useful info about the data
    # print(rs_prf.get_info('eid'))
    # print(rs_prf.get_info('rid'))
    # print(rs_prf.get_info('mid'))
    # print(rs_prf.get_info())
    #
    # # Load GDPs for temperature, including all the errors
    # gdp_prfs = MultiGDPProfile()
    # gdp_prfs.load_from_db(filt_gdp_dt, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1',
    #                       ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t',
    #                       inplace=True)
    #
    # # Let us inspect the profiles with dedicated plots.
    # gdp_prfs.plot(fn_prefix='01') # Defaults behavior, just adding a prefix to the filename.
    # gdp_prfs.plot(uc='tot', show_plt=True, fmts=[]) # Now with errors. Show it but don't save it.

    # Load a basic time profile, with a variable and altitude
    rs_prf = MultiRSProfile()
    rs_prf.load_from_db(filt_dt, 'trepros1', 'tdtpros1', alt_abbr='altpros1')
    rs_prf.sort()
    rs_prf.save_to_db()

    # Acccess some useful info about the data
    print(rs_prf.get_info('evt_id'))
    print(rs_prf.get_info('rig_id'))
    print(rs_prf.get_info('mdl_id'))
    #print(rs_prf.get_info())

    # Load GDPs for temperature, including all the errors
    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(filt_gdp_dt, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1',
                          ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t',
                          inplace=True)

    # Let us inspect the profiles with dedicated plots.
    gdp_prfs.plot(fn_prefix='01') # Defaults behavior, just adding a prefix to the filename.
    gdp_prfs.plot(uc='tot', show_plt=True, fmts=[]) # Now with errors. Show it but don't save it.
