"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

from pathlib import Path

# Set the data path to look where we currently are
from dvas.environ import path_var as dvas_path_var

# WARNING: this must be done BEFORE importing anything else ... !
dvas_path_var.orig_data_path = Path(__file__).parent / 'data'
dvas_path_var.config_dir_path = Path(__file__).parent / 'config'

# Import stuff from dvas
import dvas.plots.utils as dpu
from dvas.dvas import start_log
from dvas.data.data import MultiProfile, MultiRSProfile, MultiGDPProfile
from dvas.data.io import update_db
from dvas.database.database import DatabaseManager

# Import high-level stuff
from dvas_recipes.uaii22 import sync as drs

if __name__ == '__main__':

    # --- GENERAL SETUP ---
    # Start the logging
    start_log(2) # 1 = log to file only, 2 = file+ screen, 3 = screen only.

    # Let us fine-tune the plotting behavior of dvas
    dpu.set_mplstyle('nolatex') # The safe option. Use 'latex' fo prettier plots.

    # The generic formats to save the plots in
    dpu.PLOT_FMTS = ['pdf']

    # Show the plots on-screen ?
    dpu.PLOT_SHOW = False

    # Reset the DB to "start fresh" ?
    RESET_DB = True

    # --- DB SETUP ---

    # Create the dvas database
    db_mngr = DatabaseManager(reset_db=RESET_DB)

    # Fill the database
    if RESET_DB:
        update_db('tdtpros1', strict=True)
        update_db('trepros1', strict=True)
        update_db('treprosu_r', strict=True)
        update_db('treprosu_s', strict=True)
        update_db('treprosu_t', strict=True)
        update_db('altpros1', strict=True)


    # --- SYNCHRONIZE PROFILES ---

    #drs.sync_flight(80611, 1)

    # Get all the profiles, to see how they look.
    #filt = "all()"
    #prfs = MultiRSProfile()
    #prfs.load_from_db(filt, 'trepros1', 'tdtpros1', alt_abbr='altpros1')

    #prfs.get_info(prm='oid')
    #prfs.get_info('tags')