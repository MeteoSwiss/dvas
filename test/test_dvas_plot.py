"""
This file contains testing classes and function for
dvas.plot module.

"""

# Import from python packages and modules
import pytest
import matplotlib.pyplot as plt

# Import from current package
from dvas.plot import PLOT_TYPES

def test_plot_types():
    """Function used to test if the default plot types are supported by the OS

    The function tests:
        - compatibility of plt.savefig file extensions with the OS

    """

    # Create a dummy figure
    fig = plt.figure()
    ok_exts = fig.canvas.get_supported_filetypes()

    assert all([dvas_ext[1:] in ok_exts.keys() for dvas_ext in PLOT_TYPES])