"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.plot.plot module.

"""


# Import from python packages and modules
import matplotlib.pyplot as plt

# Import from current package
from dvas.plot.plot_utils import PLOT_TYPES


def test_plot_types():
    """Function used to test if the default plot types are supported by the OS

    The function tests:
        - compatibility of plt.savefig file extensions with the OS

    """

    # Inspired from
    # https://stackoverflow.com/questions/7608066/
    # in-matplotlib-is-there-a-way-to-know-the-list-of-available-output-format
    fig = plt.figure()
    ok_exts = fig.canvas.get_supported_filetypes()

    assert all([dvas_ext[1:] in ok_exts.keys() for dvas_ext in PLOT_TYPES])
