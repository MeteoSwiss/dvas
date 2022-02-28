"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: demo code that illustrates the core dvas functionalities
"""

# Import from Python
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Import from dvas
from dvas.dvas import Log
from dvas.dvas import Database as DB
import dvas.plots.utils as dpu
from dvas.data.data import MultiProfile, MultiRSProfile, MultiGDPProfile, MultiCWSProfile
from dvas.environ import path_var
from dvas.hardcoded import FLG_INCOMPATIBLE_NAME
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
    # WARNING: to set an "absolute" path, make sure to preface it with "/", e.g.:
    # path_var.orig_data_path = Path('/Users', 'jdoe', 'dvas_devdata')
    path_var.config_dir_path = demo_file_path.parent / 'config'
    path_var.orig_data_path = demo_file_path.parent / 'data'
    path_var.local_db_path = demo_file_path.parent / 'db'
    path_var.output_path = demo_file_path.parent / 'output'

    # Start the logging
    # Output: 0 = no logs, 1 = log to file only, 2 = file + screen, 3 = screen only.
    Log.start_log(1, level='DEBUG')

    # Fine-tune the plotting behavior of dvas
    dpu.set_mplstyle('nolatex')  # The safe option. Use 'latex' fo prettier plots.

    # The generic formats to save the plots in
    dpu.PLOT_FMTS = ['png', 'pdf']

    # Show the plots on-screen ?
    dpu.PLOT_SHOW = False

    # ----------------------------------------------------------------------------------------------
    print("\n --- DATABASE SETUP ---")

    # Use this command to refresh the DB, if it exists
    DB.refresh_db()

    # Init the DB
    DB.init()

    # Fetch
    DB.fetch_raw_data(
        [
            'time', 'gph', 'gph_uct', 'gph_ucu',
            'temp', 'temp_flg', 'temp_ucr', 'temp_ucs', 'temp_uct'
        ],
        strict=True
    )

    # Use this command to explore the DB
    # DB.explore()

    # Use this commande to extract a global view of the DB
    res = DB.extract_global_view()


    # ----------------------------------------------------------------------------------------------
    print("\n --- BASIC DATA EXTRACTION ---")

    # The demo data is comprised of 2 quadruple flights on 2017-07-12 and 2017-10-24, both with:
    # 1xRS41, 1xRS92, 1xC34, 1xC50
    # For both the RS41 and the RS92, both the GDP and manufacturer data is provided.

    # Define some basic search queries
    filt_gdp = "tags('gdp')" # Shortcut: 'gdp()'
    filt_raw = "tags('raw')" # Shortcut: 'raw()'
    filt_raw_not = "not_(tags('raw'))"
    filt_all = "all()"
    filt_dt = "dt('20171024T120000Z', '==')"

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

    # ----------------------------------------------------------------------------------------------
    print("\n --- BASIC DATA EXPLORATION ---")

    # How many profiles were loaded ?
    n_prfs = len(prfs)

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

    # MultiProfiles has a var_info property to link the DataFrame columns to the actual variable
    print("\n Content of prfs.var_info['val']:\n")
    print(prfs.var_info['val'])

    # Each profile is attributed a unique "Object Identification" (oid) number, which allows to keep
    # track of profiles throughout the dvas analysis.
    # If two profiles have the same oid, it implies that they have been acquired with the same
    # sonde AND pre-processed by the same software/recipe/etc ...
    # Note: dvas processing steps DO NOT modify the oid values.
    gdp_prf_oids = prfs.get_info('oid')

    # Flags can be used to mark specific profile elements. The possible flags, for a given
    # Profile, are accessed as follows:
    print('\nFlag ids and associated meaning:')
    for (flg_name, item) in prfs[0].flg_names.items():
        print(f"  {flg_name}: {item['flg_desc']}")

    # To flag specific elements of a given profile, use the internal methods:
    prfs[0].set_flg('user_qc', True, index=pd.Index([0, 1, 2]))

    # Let's check to see that the data was actually flagged
    print('\nDid I flag only the first three steps with "user_qc" ?')
    print(prfs[0].has_flg('user_qc'))

    # FLags are used that characterize individual measurments. Tags, on the other hand, are used to
    # characterize entire Profiles. They are useful, for example, to identify if a Profile has been
    # synchronized (tags: 'sync'), if the data is still raw (tag:'raw'), or if it belongs to a GDP
    # (tag: 'gdp'). As an example, let's figure out which Profile in rs_prfs belongs to a GDP:
    print('\nChecking Profile tags:')
    print(rs_prfs.has_tag('gdp'))

    # We can see that the different GDP profiles that were extracted and loaded (without their
    # uncertainties!) into rs_prfs were indeed correctly tagged:
    print(rs_prfs.get_info('mid'))

    # ----------------------------------------------------------------------------------------------
    print("\n --- BASIC PLOTTING ---")

    # Let us inspect the (raw) GDP profiles with dedicated plots.
    gdp_prfs.plot(fn_prefix='01-a', show=True) # Defaults behavior, just adding a prefix to the filename.

    # Repeat the same plot, but this time with the GDP uncertainties.
    # Set "show" to True to display it on-screen.
    #gdp_prfs.plot(fn_prefix='01-b', label='oid', uc='uc_tot', show=False, fmts=['png'])

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

    # Before combining the GDPs with each other, let us assess their consistency. This is a two
    # step process.
    # First, we derive inconsistencies between GDP pairs, based on a KS-test, possibly for different
    # binning values "m".
    start_time = datetime.now()
    incompat = dtgs.gdp_incompatibilities(gdp_prfs, alpha=0.0027, m_vals=[1, 6],
                                           do_plot=True, n_cpus=4)
    print('GDP mismatch derived in: {}s'.format((datetime.now()-start_time).total_seconds()))

    # Next, we derive "validities" given a specific strategy to assess the different GDP pair
    # incompatibilities ...
    valids = dtgs.gdp_validities(incompat, m_vals=[1], strategy='all-or-none')

    # ... and set them using the dvas.hardcoded.FLG_INCOMPATIBLE_NAME flag
    for gdp_prf in gdp_prfs:
        gdp_prf.set_flg(FLG_INCOMPATIBLE_NAME, True,
                        index=valids[~valids[str(gdp_prf.info.oid)]].index)

    # Let us now create a high-resolution CWS for these synchronized GDPs, making sure to drop
    # incompatible elements.
    start_time = datetime.now()
    cws = dtgg.combine(gdp_prfs, binning=1, method='weighted mean',
                       mask_flgs=FLG_INCOMPATIBLE_NAME,
                       chunk_size=150, n_cpus=8)
    print('CWS assembled in: {}s'.format((datetime.now()-start_time).total_seconds()))

    # We can now inspect the result visually
    dpg.gdps_vs_cws(gdp_prfs, cws, index_name='_idx', show=False, fn_prefix='03')

    # Save the CWS to the database.
    # One should note here that we only save the columns of the CWS DataFrame, and not the 'alt' and
    # 'tdt' indexes. As a result, if one tries to extract the cws from the DB right away, the 'alt'
    # and 'tdt' columns will be filled with NaNs.
    cws.save_to_db(add_tags=['cws'], rm_tags=['gdp'], prms=['val', 'ucr', 'ucs', 'uct', 'ucu'])

    # ----------------------------------------------------------------------------------------------
    """
    print("\n --- ASSESSMENT OF CANDIDATE RADIOSONDES ---")
    # We begin by assembling "delta profiles", i.e. deltas between candidate radiosondes and
    # associated CWS.

    # First extract all the synchronized candidate radiosonde profiles from a given flight
    filt_nogdp_dt_sync = "and_(tags('sync'), not_(gdp()), {})".format(filt_dt)
    can_prfs = MultiRSProfile()
    can_prfs.load_from_db(filt_nogdp_dt_sync, 'temp', tdt_abbr='time', alt_abbr='gph', inplace=True)

    # And now load the required CWS
    filt_cws_dt = "and_(tags('cws'), {})".format(filt_dt)
    cws_prfs = MultiCWSProfile()
    cws_prfs.load_from_db(filt_cws_dt, 'temp', tdt_abbr='time', alt_abbr='gph',
                          ucr_abbr='temp_ucr', ucs_abbr='temp_ucs', uct_abbr='temp_uct',
                          inplace=True)
    """
