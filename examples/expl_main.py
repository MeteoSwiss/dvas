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

import dvas.tools.gruan as dtg
import dvas.plots.utils as dpu


if __name__ == '__main__':


    # Let us fine-tune the plotting behavior of dvas
    dpu.set_mplstyle('nolatex') # The safe option. Use 'latex' fo prettier plots.

    # The generic formats to save the plots in
    dpu.PLOT_FMTS = ['png', 'pdf']

    # Show the plots on-screen ?
    dpu.PLOT_SHOW = False


    # Reset the DB to "start fresh" ?
    RESET_DB = True

    # Define some search queries
    filt_gdp = "tag('gdp')"
    filt_dt = "dt('20180125T120000Z', '==')"
    filt_vof = "and_(%s,%s)" % (filt_gdp, filt_dt)
    filt_gdp_der = "and_(tag('derived'), tag('gdp'))"
    filt_raw = "and_(tag('raw'), not_(tag('gdp')))"
    filt_all = "all()"
    filt_der = "and_(tag('derived'), not_(tag('gdp')))"
    filt_cws = "tag('cws')"

    # Create the database
    db_mngr = DatabaseManager(reset_db=RESET_DB)

    # Update the database + log
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
        rs_prf.load_from_db(filt_der, 'trepros1', 'tdtpros1', alt_abbr='altpros1')

    except dvasError:
        rs_prf.load_from_db(filt_raw, 'trepros1', 'tdtpros1', alt_abbr='altpros1')
        rs_prf.sort()
        #rs_prf.resample()
        rs_prf.save_to_db()

    # Acccess some useful info about the data
    print(rs_prf.get_info('evt_id'))
    print(rs_prf.get_info('rig_id'))
    print(rs_prf.get_info('mdl_id'))

    # Load GDPs for temperature, including all the errors
    gdp_prfs = MultiGDPProfile()
    gdp_prfs.load_from_db(filt_vof, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1',
                          ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t',
                          inplace=True)

    # Let us inspect the profiles with dedicated plots.
    gdp_prfs.plot(fn_prefix='01') # Defaults behavior, just adding a prefix to the filename.
    gdp_prfs.plot(uc='tot', show_plt=True, fmts=[]) # Now with errors. Show it but don't save it.


    # Compute a working standard
    # TODO: disabled in the example for now. We must first synchronize the Profiles.
    # out = dtg.combine_gdps(gdp_prfs, binning=1, method='weighted mean')
    # For the sake of the exemple, save this to the database
    #out.save_to_db()

    # And re-load it.
    #cws_prf = MultiGDPProfile()
    #cws_prf.load_from_db(filt_cws, 'trepros1', alt_abbr='altpros1', tdt_abbr='tdtpros1',
    #                     ucr_abbr='treprosu_r', ucs_abbr='treprosu_s', uct_abbr='treprosu_t',
    #                     inplace=True)
    #
    #dpg.gdps_vs_cws(gdp_prfs, cws_prf)
