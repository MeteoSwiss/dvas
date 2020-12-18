"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module content: examples
"""

from pathlib import Path

# Set the data path to look where we currently are
from dvas.environ import path_var as dvas_path_var
from dvas.environ import log_var as dvas_log_var
# WARNING: this must be done BEFORE importing anything else ... !
dvas_path_var.orig_data_path = Path(__file__).parent / 'data'
dvas_path_var.config_dir_path = Path(__file__).parent / 'config'

# Import
from dvas.data.data import MultiProfile, MultiRSProfile, MultiGDPProfile
from dvas.data.data import update_db
from dvas.database.database import DatabaseManager

from dvas.errors import dvasError
from dvas.logger import init_log as dvas_init_log

#import dvas.tools.gruan as dtg
import dvas.plots.utils as dpu


if __name__ == '__main__':

    # Set the log level we want
    dvas_log_var.log_level = 'DEBUG' # 'INFO' is the default.
    dvas_log_var.log_mode = 'CONSOLE' # 'FILE' if you want a live update

    # Start the logging
    dvas_init_log()

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
    filt_dt = "dt('20170712T000000Z', '==')"
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
        #with LogManager():
            update_db('tdtpros1', strict=True)
            update_db('trepros1', strict=True)
            update_db('treprosu_r', strict=True)
            update_db('treprosu_s', strict=True)
            update_db('treprosu_t', strict=True)
            update_db('altpros1', strict=True)