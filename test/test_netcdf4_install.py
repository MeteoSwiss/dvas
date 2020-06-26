"""
This file contains testing classes and functions for the netcdf4 module (a dvas requirement).

"""

# Import from python packages and modules
import numpy as np
import netCDF4 as nc

def test_netcd4():
    """Function used to test if the netcdf4 package is properly installed

    The function tests:
        - existence of all dependent libraries by testing the creation of a dummy netcdf4 file.

    """

    # Create a dummy file
    dummy = nc.Dataset("netcdf4_test.nc", "w", format="NETCDF4")
    dummy.description = "A dummy NETCDF4 file."

    # Include a typical dimension
    time = dummy.createDimension("time", None)

    # And a variable
    times = dummy.createVariable("time","f8",("time",))
    times.units = 's'
    times[:] = np.arange(0, 6000, 1.)

    dummy.close()

    dummy = nc.Dataset("netcdf4_test.nc", "r")

    assert dummy.data_model == 'NETCDF4'

    dummy.close()
