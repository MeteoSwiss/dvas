"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: demo code that illustrates the core dvas functionalities
"""

# Import python package
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Import stuff from dvas
from dvas.dvas import Log
from dvas.dvas import Database as DB
import dvas.plots.utils as dpu
from dvas.data.data import MultiProfile, MultiRSProfile, MultiGDPProfile
from dvas.environ import path_var
from dvas.tools import sync as dts
from dvas.tools.gdps import gdps as dtgg
from dvas.tools.gdps import stats as dtgs
import dvas.plots.gdps as dpg

# Extract our current location
demo_file_path = Path(__file__).resolve()

if __name__ == '__main__':

    print('\nStarting the dvas demonstration recipe ...\n')
    print('To understand what happens next, look at: {}'.format(Path(__file__).name))

    # ----------------------------------------------------------------------------------------------
    print("\n --- GENERAL SETUP ---")

    # Init paths
    path_var.config_dir_path = demo_file_path.parent / 'config'
    path_var.orig_data_path = Path('/Users', 'fvogt', 'Projects', 'MCH', 'UAII2021', 'dvas',
                                   'codedev', 'testzone', 'dvas_devdata', 'devdata', 'data')
    #path_var.orig_data_path = Path('TEST')
    path_var.local_db_path = demo_file_path.parent / 'db'
    path_var.output_path = demo_file_path.parent / 'output'

    # Start the logging
    Log.start_log(2, level='DEBUG')  # 0 = no logs, 1 = log to file only, 2 = file + screen, 3 = screen only.

    # Fine-tune the plotting behavior of dvas
    dpu.set_mplstyle('nolatex')  # The safe option. Use 'latex' fo prettier plots.

    # The generic formats to save the plots in
    dpu.PLOT_FMTS = ['png', 'pdf']

    # Show the plots on-screen ?
    dpu.PLOT_SHOW = False

    # ----------------------------------------------------------------------------------------------
    print("\n --- DATABASE SETUP ---")

    start_time = datetime.now()

    # Use this command to clear the DB
    DB.clear_db()

    # Init the DB
    DB.init()

    # Fetch
    DB.fetch_raw_data(
        [
            'time', 'gph',
            'temp', 'temp_flag', 'temp_ucr', 'temp_ucs', 'temp_uct',
        ],
        strict=True
    )

    print('Database setup in: {}s'.format((datetime.now()-start_time).total_seconds()))

    # Use this command to explore the DB
    # DB.explore()

    # ----------------------------------------------------------------------------------------------
    print("\n --- BASIC DATA EXTRACTION ---")

    # Define some basic search queries
    filt_gdp = "tags('gdp')"
    filt_raw = "tags('raw')"
    filt_raw_not = "not_(tags('raw'))"
    filt_all = "all()"
    filt_dt = "dt('20210319T090000Z', '==')"

    # Define some more complex queries
    filt_raw_dt = "and_({}, {})".format(filt_raw, filt_dt)
    filt_raw_gdp_dt = "and_({}, {}, {})".format(filt_raw, filt_gdp, filt_dt)

    # Load a series of basic profiles associated to a specific set of search criteria.
    # Each profile consists of a variable and an associated altitude.
    prfs = MultiProfile()
    prfs.load_from_db(filt_raw, 'temp', 'gph')

    # Idem for a series of radiosonde profiles, consisting of a variable, an associated timestep,
    # and an altitude.
    rs_prfs = MultiRSProfile()
    rs_prfs.load_from_db(filt_raw_dt, 'temp', 'time', alt_abbr='gph')

    # Load GDPs for temperature, including all the errors at hand
    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(filt_raw_gdp_dt, 'temp', tdt_abbr='time', alt_abbr='gph',
                          ucr_abbr='temp_ucr', ucs_abbr='temp_ucs', uct_abbr='temp_uct',
                          inplace=True)

    """
    # ----------------------------------------------------------------------------------------------
    print("\n --- BASIC DATA EXPLORATION ---")

    # How many profiles were loaded ?
    n_prfs = len(prfs)

    # What data has been lodaed from the DB ?
    print("\nMulti-profile data content:\n")
    print(prfs.var_info)

    # Each Profile carries an InfoManager entity with it, which contains useful data.
    # Side note: MultiProfile entities are iterable !
    print('\nContent of a profile InfoManager:\n')
    print(prfs[0].info)

    # How many distinct events are present in prfs ?
    prfs_evts = set(prfs.get_info('eid'))

    # The data is stored inside Pandas dataframes. Each type of profile contains a different set of
    # columns and indexes.
    prf_df = prfs[0].data
    print('\nBasic profile dataframe:\n  index.names={}, columns={}'.format(prf_df.index.names,
                                                                            prf_df.columns.to_list()))
    rs_prf_df = rs_prfs[0].data
    print('\nRS profile dataframe:\n  index.names={}, columns={}'.format(rs_prf_df.index.names,
                                                                       rs_prf_df.columns.to_list()))
    gdp_prf_df = gdp_prfs[0].data
    print('\nGDP profile dataframe:\n  index.names={}, columns={}'.format(gdp_prf_df.index.names,
                                                                        gdp_prf_df.columns.to_list()))

    # Each profile is attributed a unique "Object Identification" (oid) number, which allows to keep
    # track of profiles throughout the dvas analysis.
    # If two profiles have the same oid, it implies that they have been acquired with the same
    # sonde AND pre-processed by the same software/recipe/etc ...
    # Note: dvas processing steps DO NOT modify the oid values.
    gdp_prf_oids = prfs.get_info('oid')

    # Flags can be used to mark specific profile elements. The possible flags, for a given
    # Profile, are accessed as follows:
    print('\nFlag ids and associated meaning:')
    print(prfs[0].flags_name)

    # To flag specific elements of a given profiles, use the internal methods:
    prfs[0].set_flg('user_qc', True, index=pd.Index([0, 1, 2]))

    # Let's check to see that the data was actually flagged
    print('\nDid I flag only the first three steps with "user_qc" ?')
    print(prfs[0].is_flagged('user_qc'))

    # ----------------------------------------------------------------------------------------------
    print("\n --- BASIC PLOTTING ---")

    # Let us inspect the (raw) GDP profiles with dedicated plots.
    gdp_prfs.plot(fn_prefix='01') # Defaults behavior, just adding a prefix to the filename.
    # Now with errors. Show the plot but don't save it.
    gdp_prfs.plot(label='oid', uc='uc_tot', show=True, fmts=[])

    # ----------------------------------------------------------------------------------------------
    print("\n --- PROFILE RESAMPLING ---")

    # The RS-92 GDP is not being issued on a regular grid. Let's resample it.
    gdp_prfs_1s = gdp_prfs.resample(freq='1s', inplace=False)

    # We can now save the modified Profiles into the database, with a suitable tag to identify them.
    gdp_prfs_1s.save_to_db(add_tags=['1s'])

    # ----------------------------------------------------------------------------------------------
    print("\n --- PROFILE SYNCHRONIZATION ---")

    # Most likely, distinct profiles will have distinct lengths
    print("\nGDP lengths pre-synchronization: ", [len(item.data) for item in gdp_prfs_1s])

    # Profiles can be synchronized using tools located uner dvas.tools.sync

    # Synchronizing profiles is a 2-step process. First, the shifts must be identified.
    # dvas contains several routines to do that under dvas.tools.sync
    # For example, the most basic one is to compare the altitude arrays
    gdp_prfs_1s.sort() # <- This helps keep the order of Profiles consistent between runs.
    sync_shifts = dts.get_sync_shifts_from_alt(gdp_prfs_1s)

    # A fancier option is to look at the profile values, and minimize the mean of their absolute
    # difference
    #sync_shifts = dts.get_sync_shifts_from_val(gdp_prfs, max_shift=50, first_guess=sync_shifts)

    # Given these shifts, let's compute the new length of the synchronized Profiles.
    # Do it such that no data is actually cropped out, i.e. add NaN/NaT wherever needed.
    raw_lengths = [len(item.data) for item in gdp_prfs_1s.profiles]
    sync_length = np.max(np.array(sync_shifts) + np.array(raw_lengths)) - np.min(sync_shifts)

    # Once a set of shifts has been identified, they can be applied
    gdp_prfs_1s.rebase(sync_length, shifts=sync_shifts)

    # We can see that these have indeed been synchronized because all the profiles have the same
    # length. Note that because the 'alt' and 'tdt' index are **also** shifted as part of the
    # synchronization, that step is not immediately visible in the plots.
    print("\nGDP lengths post-synchronization: ", [len(item.data) for item in gdp_prfs_1s])

    # Save the synchronized profiles to the DB, adding the 'sync' tag for easy identification.
    gdp_prfs_1s.save_to_db(add_tags=['sync'])

    # ----------------------------------------------------------------------------------------------
    print("\n --- ASSEMBLY OF A COMBINED WORKING STANDARD ---")

    # If GDPs are synchronized, they can be combined into a Combined Working Standard (CWS) using
    # tools located inside dvas.tools.gdps

    # Let us begin by extracting the synchronized GDPs for a specific flight
    filt_gdp_dt_sync = "and_(tags('sync'), {}, {})".format(filt_gdp, filt_dt)

    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(filt_gdp_dt_sync, 'temp', tdt_abbr='time', alt_abbr='gph',
                          ucr_abbr='temp_ucr', ucs_abbr='temp_ucs', uct_abbr='temp_uct',
                          inplace=True)

    # Before combining the GDPs with each other, let us assess their consistency. The idea here is
    # to flag any inconsistent measurement, so that they can be ignored during the combination
    # process.
    start_time = datetime.now()
    out = dtgs.get_incompatibility(gdp_prfs, alpha=0.0027, bin_sizes=[1, 2, 4, 8], do_plot=True,
                                   n_cpus=4)
    print('GDP mismatch derived in: {}s'.format((datetime.now()-start_time).total_seconds()))

    # TODO: set flags based on the incompatibilities derived.

    # Let us now create a high-resolution CWS for these synchronized GDPs
    start_time = datetime.now()
    cws = dtgg.combine(gdp_prfs, binning=1, method='weighted mean', chunk_size=200, n_cpus=4)
    print('CWS assembled in: {}s'.format((datetime.now()-start_time).total_seconds()))

    # We can now inspect the result visually
    dpg.gdps_vs_cws(gdp_prfs, cws, index_name='_idx', show=True, fn_prefix='03')

    # --- TODO ---
    # Save the CWS to the database
    #cws.save_to_db()

    # Let's compare this CWS with the original data
    # Let us begin by extracting the synchronized GDPs for a specific flight
    #filt_cws_gdp_dt_sync = "or_({},tags('cws'))".format(filt_gdp_dt_sync)

    #gdp_cws_prfs = MultiGDPProfile()
    #gdp_cws_prfs.load_from_db(filt_cws_gdp_dt_sync, 'trepros1', alt_abbr='altpros1',
    #                          tdt_abbr='tdtpros1',
    #                          ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t',
    #                          inplace=True)
    """
