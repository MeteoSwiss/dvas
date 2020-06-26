"""
This file contains testing classes and functions for the netcdf4 module (a dvas requirement).

"""

# Import from python packages and modules
import netcdf4 as nc

def test_netcd4():
    """Function used to test if the netcdf4 package is properly installed

    The function tests:
        - existence of all dependent libraries by testing the creation of a dummy netcdf4 file.

    """

    dummy = Dataset("netcdf4_test.nc", "w", format="NETCDF4")
    dummy.close()

    dummy = Dataset("netcdf4_test.nc", "r")

    assert dummy.data_model == 'NETCDF4'

    dummy.close()
