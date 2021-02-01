"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: examples
"""

from pathlib import Path

# Set the data path to look where we currently are
from dvas.environ import path_var as dvas_path_var

# WARNING: this must be done BEFORE importing anything else ... !
dvas_path_var.orig_data_path = Path(__file__).parent / 'data'
dvas_path_var.config_dir_path = Path(__file__).parent / 'config'

# Import
from dvas.data.data import MultiProfile, MultiRSProfile, MultiGDPProfile
from dvas.data.io import update_db
from dvas.database.database import DatabaseManager

import dvas.plots.utils as dpu

from dvas.dvas import start_log

if __name__ == '__main__':

    # Start the logging
    start_log(2)

    # Let us fine-tune the plotting behavior of dvas
    dpu.set_mplstyle('nolatex') # The safe option. Use 'latex' fo prettier plots.

    # The generic formats to save the plots in
    dpu.PLOT_FMTS = ['png', 'pdf']

    # Show the plots on-screen ?
    dpu.PLOT_SHOW = True

    # Reset the DB to "start fresh" ?
    RESET_DB = True

    # Define some basic, self-explanatory, search queries
    filt_gdp = "tag('gdp')"
    filt_evt = "tag('e:80596')"
    filt_rig = "tag('r:1')"
    filt_raw = "tag('raw')"
    filt_dt = "dt('20171024T120000Z', '==')"

    # Define advanced search queries
    # 1) Anything that is NOT raw data
    filt_not_raw = "not_(tag('gdp'))"
    # 2) Raw data for a specific flight (i.e. event and rig)
    filt_flight = "and_(%s, %s, %s)" % (filt_evt, filt_rig, filt_raw)
    # 3) Raw GDPs for a specific datetime of launch
    filt_dt_gdps = "and_(%s, %s, %s)" % (filt_gdp, filt_dt, filt_raw)

    # Create the database
    db_mngr = DatabaseManager(reset_db=RESET_DB)

    # Update the database + log
    if RESET_DB:
        update_db('tdtpros1', strict=True)
        update_db('trepros1', strict=True)
        update_db('treprosu_r', strict=True)
        update_db('treprosu_s', strict=True)
        update_db('treprosu_t', strict=True)
        update_db('altpros1', strict=True)

    # Load a basic profile, with a variable, and altitude.
    prf = MultiProfile()
    prf.load_from_db(filt_flight, 'trepros1', 'altpros1')

    # Load a basic time profile, with a variable and altitude
    rs_prf = MultiRSProfile()
    rs_prf.load_from_db(filt_flight, 'trepros1', 'tdtpros1', alt_abbr='altpros1')

    # Acccess some useful info about the data
    print(rs_prf.get_info('evt_id'))
    print(rs_prf.get_info('rig_id'))
    print(rs_prf.get_info('mdl_id'))
    print(rs_prf.get_info())

    # Load GDPs for temperature, including all the errors
    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(filt_dt_gdps, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1',
                          ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t',
                          ucu_abbr='treprosu_u',
                          inplace=True)

    # Let us inspect the profiles with dedicated plots.
    gdp_prfs.plot(fn_prefix='01') # Defaults behavior, just adding a prefix to the filename.
    gdp_prfs.plot(uc='tot', show_plt=True, fmts=[]) # Now with errors. Show it but don't save it.
