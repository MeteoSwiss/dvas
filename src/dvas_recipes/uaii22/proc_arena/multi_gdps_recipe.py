"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This recipe is used to process (pre-UAII22) multi-GDP flights. It can be expected that the final
UAII22 recipe will be strongly based on this one.
"""

# Import python package
from pathlib import Path

# Import stuff from dvas
from dvas.dvas import Database as DB

# Import high-level stuff
from dvas_recipes import utils as dru
from dvas_recipes.uaii22 import sync as drs
from dvas_recipes.uaii22 import plots as drp
from dvas_recipes.uaii22 import gdps as drg

# Extract our current location
rcp_fpath = Path(__file__).resolve()

if __name__ == '__main__':

    # --- GENERAL SETUP ---
    rcp_vars = dru.initialize_recipe(rcp_fpath)

    # --- DB SETUP ---
    # Reset the DB to "start fresh" ?
    RESET_DB = True

    if RESET_DB:
        # Use this command to clear the DB
        DB.clear_db()

        # Init the DB
        DB.init()

        # Fetch
        DB.fetch_raw_data(['time', 'gph'] +
                          list(rcp_vars) +
                          [rcp_vars[var][uc] for var in rcp_vars for uc in rcp_vars[var]],
                          strict=True)

    # What are the flights id of interest ?
    eids = [139164, # day, RS-41 x2, M10 x1
            #139165, # day, RS-41 x1, M10 x1
            #139926, # day, RS-41 x2, M10 x2,
            #140075, # night, RS-41 x2, M10 x1
            ]

    # --- SYNCHRONIZE PROFILES ---
    for eid in eids:
        drs.sync_flight(eid, 1, rcp_vars)

    # --- GENERIC PROFILE PLOT ---
    # Make a plot showing all the variables of interest.
        drp.flight_overview(eid, 1, rcp_vars, tags='sync', step_id='01b')

    # --- ASSEMBLE GDPS ---

    #drg.build_cws(80611, 1)


    # Get all the profiles, to see how they look.
    #filt = "all()"
    #prfs = MultiRSProfile()
    #prfs.load_from_db(filt, 'temp', 'time', alt_abbr='gph')

    #prfs.get_info(prm='oid')
    #prfs.get_info('tags')
