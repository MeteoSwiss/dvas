"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Resample strategies

"""

# Import from external packages
import pandas as pd

# Import from current package
from ...errors import DvasError
from .data import MPStrategyAC
from ...hardcoded import PRF_REF_INDEX_NAME, PRF_REF_TDT_NAME

class ResampleRSStrategy(MPStrategyAC):
    """Class to manage the (time) resampling of Profiles"""

    def execute(self, prfs, freq='1s'):
        """Implementation of time resampling method for Profiles.

        Args:
            prfs (list of RSProfiles|GDPProfiles): Profiles to resample.
            freq (str, optional): see pandas.timedelta_range(). Defaults to '1s'.

        Returns:
            dvas.data.MultiRSProfile|MultiGDPProfile: the resampled MultiProfile.

        """

        # Some sanity checks to begin with
        if not isinstance(prfs, list):
            raise DvasError("Ouch ! prfs should be of type list, and not: {}".format(type(prfs)))
        # The following should in principle never happen because the strategy ensures that.
        # If this blows up, then something must have gone really wrong ...
        if any([PRF_REF_TDT_NAME not in prf.get_index_attr() for prf in prfs]):
            raise DvasError("Ouch ! I can only resample profiles with a timedelta array ...")

        # Very well, let's start looping and resampling each Profile
        for (prf_ind, prf) in enumerate(prfs):

            # Let's identify the min and max integer values, rounded to the nearest second.
            t0 = min(prf.data.index.get_level_values(PRF_REF_TDT_NAME)).ceil('1s')
            t1 = max(prf.data.index.get_level_values(PRF_REF_TDT_NAME)).ceil('1s')
            # Turn this into a regular grid
            new_tdt = pd.timedelta_range(t0, t1, freq=freq, name=PRF_REF_TDT_NAME)

            # Check if the new index is the same as the old. And if so, continue to the next
            # next profile without changing anything
            if len(new_tdt)==len(prf.data):
                if all(new_tdt == prf.data.index.get_level_values(PRF_REF_TDT_NAME)):
                    continue

            # Get ready to interpolate.
            # First, get the original data out, reseting all the indices.
            this_data = prf.data.reset_index()

            # Let's drop the original integer index, to avoid type conversion issues
            this_data.drop(columns=PRF_REF_INDEX_NAME, inplace=True)

            # Create a new dataframe to keep the interpolated stuff
            new_data = pd.DataFrame(new_tdt, columns=this_data.columns)

            # Now, start looping through each layer. This may seem dumb (and probably is), but it
            # remains faster than propagating all the errors in one go with a massive matrix.
            for ind in range(len(new_data)):

                # Where do I want to interpolate stuff onto ?
                xi_prime = new_tdt[ind]

                # Identify the nearest surrounding points
                o_ind = this_data[PRF_REF_TDT_NAME].searchsorted(xi_prime)
                xi = this_data[PRF_REF_TDT_NAME][o_ind-1]
                xi_p1 = this_data[PRF_REF_TDT_NAME][o_ind]

                fact = (xi_prime-xi)/(xi_p1-xi)

                # Perform the interolation
                for name in this_data.columns:
                    if name == PRF_REF_TDT_NAME:
                        continue

                    new_data[name][ind] = fac*this_data[name][o_ind] + \
                                          (1-fac) * this_data[name][o_ind-1]

            # And finally let's assign the new DataFrame to the Profile. The underlying setter
            # will take care of reformatting all the indices as needed.
            prfs[prf_ind].data = new_data

            return prfs

class ResampleGDPStrategy(MPStrategyAC):
    """ Class to handle the resample strategy for GDPs, incl.full error propagation.

    """

    
    #TODO
