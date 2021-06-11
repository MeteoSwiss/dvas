"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.plots.utils module.

"""


# Import from python packages and modules
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import from current package
import dvas.plots.utils as dpu

def test_plot_fmts():
    """ Function used to test if the default plot formats are supported by the OS

    The function tests:
        - compatibility of plt.savefig file extensions with the OS

    """

    # Inspired from
    # https://stackoverflow.com/questions/7608066/
    # in-matplotlib-is-there-a-way-to-know-the-list-of-available-output-format
    fig = plt.figure()
    ok_exts = fig.canvas.get_supported_filetypes()

    assert all([dvas_ext in ok_exts.keys() for dvas_ext in dpu.PLOT_FMTS])

def test_plot_styles():
    """ Function to test if I can properly set the dvas plotting styles.

    This function tests:
        - ability to switch between Latex and noLatex styles.
    """

    # Set the Latex style
    dpu.set_mplstyle('latex')
    assert mpl.rcParams['text.usetex']

    # Undo it
    dpu.set_mplstyle('nolatex')
    assert not mpl.rcParams['text.usetex']

def test_fix_txt():
    """ Function to test the correction of strings for plots, depending on the chosen style.
    """

    # Set the LateX style
    dpu.set_mplstyle('latex')
    assert dpu.fix_txt('_idx') == r'\_idx'
    assert dpu.fix_txt('[%]') == r'[\%]'
