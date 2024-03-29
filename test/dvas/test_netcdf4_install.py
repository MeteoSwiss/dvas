# -*- coding: utf-8 -*-
# pylint: disable= W0612
"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function of the netCDF4 dependency.

"""

# Import from python packages and modules
import os
import numpy as np
import netCDF4 as nc

def test_netcd4():
    """Function used to test if the netcdf4 package is properly installed.

    The function tests:
        - existence of all dependent libraries by testing the creation of a dummy netcdf4 file.

    """

    # Create a dummy file
    dummy = nc.Dataset("netcdf4_test.nc", "w", format="NETCDF4")
    dummy.description = "A dummy NETCDF4 file."

    # Create a group (special feature of netCDF4)
    dummy_grp = dummy.createGroup("dummy_grp")

    # Include a typical dimension
    time = dummy_grp.createDimension("time", None)

    # And a variable
    times = dummy_grp.createVariable("time", "f8", ("time",), zlib=True, least_significant_digit=3)
    times.units = 's'
    times[:] = np.arange(0, 6000, 1.5)

    dummy.close()

    dummy = nc.Dataset("netcdf4_test.nc", "r")

    # If I get here, then most likely it is all working fine.
    assert dummy.data_model == 'NETCDF4'

    # Clean up the mess I made.
    dummy.close()
    os.remove("netcdf4_test.nc")
