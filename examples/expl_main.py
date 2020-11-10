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

if __name__ == '__main__':

    # Define
    RESET_DB = False

    filt_gdp = "tag('gdp')"
    filt_raw = "tag('raw')"
    filt_dt = "dt('20160715T120000Z', '==')"

    filt_in = "and_(%s,%s,%s)" % (filt_gdp, filt_raw, filt_dt)
    filt_ws = "and_(%s,not_(%s),%s)" % (filt_gdp, filt_raw, filt_dt)
    filt_all = "and_(%s,%s)" % (filt_gdp, filt_dt)

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

    # Load a basic profile, with a variable, and altitude.
    prf = MultiProfile()
    prf.load(filt_in, 'trepros1', alt_abbr='altpros1', inplace=True)

    # Make a plot
    prf.plot(fig_num=1, save_fn='plot1')

    # Load a regular radiosonde profile, with a variable, altitude, and time deltas.
    rs_prf = MultiRSProfile()
    rs_prf.load(filt_in, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1', inplace=True)

    # Load the GDPs for temperature, including all the errors
    gdp_prf = MultiGDPProfile()
    gdp_prf.load(filt_in, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1',
                 ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t', inplace=True)

    # Use convenience getters to extract some info
    srns = gdp_prf.get_evt_prm('sn')
    vals_alt = gdp_prf.get_prms(['val', 'alt'])

    # Compute the total error from GDPs
    uc_tot = gdp_prf.uc_tot



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
