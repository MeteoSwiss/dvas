"""
Copyright(c) 2020 MeteoSwiss, contributors listed in AUTHORS

Distributed under the terms of the BSD 3 - Clause License.

SPDX - License - Identifier: BSD - 3 - Clause

Module contents: Testing classes and function for dvas.plot.plot module.

"""


# Import from python packages and modules
import matplotlib.pyplot as plt

# Import from current package
from dvas.plot import PLOT_TYPES


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
