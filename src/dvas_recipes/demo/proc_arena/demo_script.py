"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: demo code that illustres the core dvas functionalities
"""

# Import python package
from pathlib import Path
import numpy as np

# Import stuff from dvas
import dvas.plots.utils as dpu
from dvas.dvas import start_log
from dvas.data.data import MultiProfile
from dvas.data.io import update_db
from dvas.database.database import DatabaseManager
from dvas.environ import path_var
from dvas.tools import sync as dts
from dvas.tools import gruan as dtg

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
    dpu.set_mplstyle('latex') # The safe option. Use 'latex' fo prettier plots.

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

    # --- BASIC DATA EXTRACTION ---

    # Define some basic search queries
    filt_gdp = "tags('gdp')"
    filt_raw = "tags('raw')"
    filt_raw_not = "not_(tags('raw'))"
    filt_all = "all()"
    filt_dt = "dt('20171024T120000Z', '==')"

    # Define some more complex queries
    filt_raw_dt = "and_({}, {})".format(filt_raw, filt_dt)
    filt_raw_gdp_dt = "and_({}, {}, {})".format(filt_raw, filt_gdp, filt_dt)

    # Add a flag
    prf.profiles[0].set_flg('raw_na', True)

    # Load a basic time profile, with a variable and altitude
    rs_prf = MultiRSProfile()
    rs_prf.load_from_db(filt_dt, 'trepros1', 'tdtpros1', alt_abbr='altpros1')
    rs_prf.sort()
    rs_prf.save_to_db()

    # Idem for a series of radiosonde profiles, consisting of a variable and an associated timestep
    # and altitude.
    rs_prfs = MultiRSProfile()
    rs_prfs.load_from_db(filt_raw_dt, 'trepros1', 'tdtpros1', alt_abbr='altpros1')

    # Load GDPs for temperature, including all the errors
    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(filt_raw_gdp_dt, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1',
                          ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t',
                          inplace=True)

    # --- BASIC DATA EXPLORATION ---

    # How many profiles were loaded ?
    n_prfs = len(prfs)

    # Each Profile carries an InfoManager entity with it, which contains useful data:
    print('\nContent of a profile InfoManager:')
    print(prfs.info[0])

    # How many distinct events are present in prfs ?
    prfs_evts = set(prfs.get_info('evt_id'))

    # The data is stored inside Pandas dataframes. Each type of profile contains a different set of
    # columns and indexes.
    prf_df = prfs.profiles[0].data
    print('\nBasic profile dataframe:\n  index.names={}, columns={}'.format(prf_df.index.names,
                                                                            prf_df.columns))
    rs_prf_df = rs_prfs.profiles[0].data
    print('RS profile dataframe:\n  index.names={}, columns={}'.format(rs_prf_df.index.names,
                                                                       rs_prf_df.columns))
    gdp_prf_df = gdp_prfs.profiles[0].data
    print('GDP profile dataframe:\n  index.names={}, columns={}'.format(gdp_prf_df.index.names,
                                                                        gdp_prf_df.columns))

    # Each profile is attributed a unique "Object Identification" (oid) number, which allows to keep
    # track of them throughout the dvas analysis.
    # If two profiles have the same oid, it implies that they have been acquired with the same
    # sonde, and pre-processed by the same software/recipe/etc ...
    # Note: dvas processing steps DO NOT modify the oid values.
    gdp_prf_oids = prfs.get_info('oid')

    # --- BASIC PLOTTING ---

    # Let us inspect the (raw) GDP profiles with dedicated plots.
    gdp_prfs.plot(fn_prefix='01') # Defaults behavior, just adding a prefix to the filename.
    gdp_prfs.plot(uc='tot', show_plt=True, fmts=[]) # Now with errors. Show it but don't save it.

    # --- PROFILE SYNCHRONIZATION ---

    # Synchronizing profiles is a 2-step process. First, the shifts must be identified.
    # dvas contains several routines to do that under dvas.tools.sync
    # For example, the most basic one is to compare the raw altitude arrays
    gdp_prfs.sort() # <- This helps keep the order straight.
    sync_shifts = dts.get_sync_shifts_from_alt(gdp_prfs)

    import pdb
    pdb.set_trace()

    # A fancier option is to look at the profile values, and minimize the mean of their absolute
    # difference
    sync_shifts = dts.get_sync_shifts_from_val(gdp_prfs, max_shift=50, first_guess=sync_shifts)

    # Given these shifts, let's compute the new length of the synchronized Profiles.
    # Do it such that no data is actually cropped out, i.e. add NaN/NaT wherever needed.
    raw_lengths = [len(item.data) for item in gdp_prfs.profiles]
    sync_length = np.max(np.array(sync_shifts) + np.array(raw_lengths)) - np.min(sync_shifts)

    # Once a set of shifts has been identified, they can be applied
    gdp_prfs.rebase(sync_length, shifts=sync_shifts)

    # Save the synchronized profiles to the DB, adding the 'sync' tag for easy identification.
    gdp_prfs.save_to_db(add_tags=['sync'])

    # --- ASSEMBLY OF COMBINED WORKING STANDARD ---

    # If GDPs are synchronized, they can be combined into a Combined Working Standard (CWS) using
    # tools located inside dvas.tools.gruan

    # Let us begin by extracting the synchronized GDPs for a specific flight
    filt_gdp_dt_sync = "and_(tags('sync'),{}, {})".format(filt_gdp, filt_dt)

    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(filt_gdp_dt_sync, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1',
                          ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t',
                          inplace=True)

    # We can see that these have indeed been synchronized because all the profiles have the same
    # length. Note that because the 'alt' and 'tdt' index are **also** shifted as part of the
    # synchronization, that step is not immediately visible in the plots.
    print("\nGDP lengths post-synchronization: ", [len(item) for item in gdp_prfs.get_prms()])

    # Let us now create a high-resolution CWS for these synchronized GDPs
    cws = dtg.combine_gdps(gdp_prfs, binning=1, method='weighted mean')
    cws.save_to_db()

    # Let's compare this CWS with the original data
    # Let us begin by extracting the synchronized GDPs for a specific flight
    filt_cws_gdp_dt_sync = "or_({},tags('cws'))".format(filt_gdp_dt_sync)

    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(filt_cws_gdp_dt_sync, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1',
                          ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t',
                          inplace=True)
