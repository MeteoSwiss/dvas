"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

# Import python package
from pathlib import Path

# Import stuff from dvas
from dvas.dvas import Log
from dvas.dvas import Database as DB
import dvas.plots.utils as dpu
from dvas.data.data import MultiRSProfile
from dvas.environ import path_var

# Import high-level stuff
from dvas_recipes.uaii22 import sync as drs
from dvas_recipes.uaii22 import gdps as drg

# Extract our current location
demo_file_path = Path(__file__).resolve()

if __name__ == '__main__':

    # --- GENERAL SETUP ---

    # Init paths
    path_var.config_dir_path = demo_file_path.parent / 'config'
    path_var.orig_data_path = demo_file_path.parent / 'data'
    path_var.local_db_path = demo_file_path.parent / 'db'
    path_var.output_path = demo_file_path.parent / 'output'

    # Start the logging
    Log.start_log(2) # 1 = log to file only, 2 = file+ screen, 3 = screen only.

    # Let us fine-tune the plotting behavior of dvas
    dpu.set_mplstyle('nolatex') # The safe option. Use 'latex' fo prettier plots.

    # The generic formats to save the plots in
    dpu.PLOT_FMTS = ['pdf']

    # Show the plots on-screen ?
    dpu.PLOT_SHOW = False

    # Reset the DB to "start fresh" ?
    RESET_DB = True

    # --- DB SETUP ---

    # Use this command to clear the DB
    DB.clear_db()

    # Init the DB
    DB.init()

    # Fetch
    DB.fetch_raw_data(
        [
            'tdtpros1',
            'trepros1',
            'treprosu_r', 'treprosu_s', 'treprosu_t',
            'altpros1'
        ],
        strict=True
    )

    # --- SYNCHRONIZE PROFILES ---

    drs.sync_flight(80611, 1)

    # --- ASSEMBLE GDPS ---

    drg.build_cws(80611, 1)


    # Get all the profiles, to see how they look.
    filt = "all()"
    prfs = MultiRSProfile()
    prfs.load_from_db(filt, 'trepros1', 'tdtpros1', alt_abbr='altpros1')

    #prfs.get_info(prm='oid')
    #prfs.get_info('tags')
