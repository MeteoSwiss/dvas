"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Resample strategies

"""

# Import from external packages
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

# Import from current package
from ...logger import data as logger
from ...errors import DvasError
from .data import MPStrategyAC
from ...tools.gdps.correlations import coeffs
from ...hardcoded import PRF_REF_INDEX_NAME, PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_FLG_NAME
from ...hardcoded import PRF_REF_VAL_NAME, PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME
from ...hardcoded import PRF_REF_UCU_NAME

class ResampleStrategy(MPStrategyAC):
    """ Class to handle the resample strategy for RS and GDPs Profiles. """

    def execute(self, prfs, freq='1s'):
        """Implementation of time resampling method for RS and GDP Profiles.

        .. note::

            This strategy does NOT treat NaN's in a special way. This implies that if a NaN is one
            of the two closest original data points from a new location to be interpolated, that
            location will result in a NaN as well.

        Args:
            prfs (list of RSProfiles|GDPProfiles): GDP Profiles to resample.
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
            t_0 = min(prf.data.index.get_level_values(PRF_REF_TDT_NAME)).ceil('1s')
            t_1 = max(prf.data.index.get_level_values(PRF_REF_TDT_NAME)).floor('1s')

            # Turn this into a regular grid
            new_tdt = pd.timedelta_range(t_0, t_1, freq=freq, name=PRF_REF_TDT_NAME)

            # Get the existing time steps
            old_tdt = prf.data.index.get_level_values(PRF_REF_TDT_NAME)

            # Check if the new index is the same as the old. And if so, continue to the next
            # next profile without changing anything
            if len(new_tdt) == len(prf.data):
                if all(new_tdt == prf.data.index.get_level_values(PRF_REF_TDT_NAME)):
                    continue

            # Get ready to interpolate.
            # First, get the original data out, reseting all the indices to columns.
            this_data = prf.data.reset_index()

            # Let's drop the original integer index, to avoid type conversion issues
            this_data.drop(columns=PRF_REF_INDEX_NAME, inplace=True)

            # What are the indices of the closest (upper) points for each (new) step ?
            x_ip1_ind = [this_data[PRF_REF_TDT_NAME].searchsorted(xi_star,
                         side='right') for xi_star in new_tdt]

            # ---- This is useful only for GDP Profiles ----
            # Compute the Jacobian matrix
            omega_vals = [(item-old_tdt.values[x_ip1_ind[ind]-1])/
                          np.diff(old_tdt.values)[x_ip1_ind[ind]-1]
                          for (ind, item) in enumerate(new_tdt.values)]

            # Create the G matrix to propagate errors
            G_mat = np.zeros((len(new_tdt), len(old_tdt)))

            # Fill it with the appropriate values.
            # This is not particularly smart, not fast. Could I do better ?
            for (ind, val) in enumerate(omega_vals):
                G_mat[ind][x_ip1_ind[ind]-1:x_ip1_ind[ind]+1] = np.array([1-val, val])
            # ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

            # Create a new dataframe to keep the interpolated stuff
            new_data = pd.DataFrame(new_tdt, columns=this_data.columns)

            # Loop through the different columns
            for name in this_data.columns:

                if name == PRF_REF_FLG_NAME:
                    # Flags will be set using the Profile class method further down.
                    # until then, simply copy the existing flags over.
                    new_data.loc[:, name] = this_data.loc[:, name]

                # Interpolate the data. That's the easy bit.
                if name in [PRF_REF_ALT_NAME, PRF_REF_VAL_NAME]:
                    #TODO: This should not be done via .astype, but via to_timedelta('s') !!!
                    func = interp1d(old_tdt.values.astype('int64'),
                                    this_data.loc[:, name].values, kind='linear')

                    new_data.loc[:, name] = func(new_tdt.values.astype('int64'))

                # Propagate the errors if it comes to that
                if name in [PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME]:

                    # Let's compute the covariance matrix
                    # Here it doesn't matter what the oid, rid, eid, mid actually are. It's just
                    # one profile so it's the same for all the points.
                    U_mat = coeffs(np.tile(range(len(old_tdt)), (len(old_tdt), 1)), # i
                                   np.tile(range(len(old_tdt)), (len(old_tdt), 1)).T, # j
                                   name,
                                   oid_i=np.ones((len(old_tdt), len(old_tdt))),
                                   oid_j=np.ones((len(old_tdt), len(old_tdt))),
                                   mid_i=np.ones((len(old_tdt), len(old_tdt))),
                                   mid_j=np.ones((len(old_tdt), len(old_tdt))),
                                   rid_i=np.ones((len(old_tdt), len(old_tdt))),
                                   rid_j=np.ones((len(old_tdt), len(old_tdt))),
                                   eid_i=np.ones((len(old_tdt), len(old_tdt))),
                                   eid_j=np.ones((len(old_tdt), len(old_tdt))),
                                  )

                    # Multiply with the errors. Mind the structure of these arrays to get the
                    # correct mix of Hadamard and dot products where I need them !
                    sigmas = np.array([this_data.loc[:, name].values])
                    U_mat = np.multiply(U_mat, sigmas.T @ sigmas)

                    # Let's compute the full covariance matrix for the interpolated profile
                    # This is a square matrix, with the off-axis elements containing the covarience
                    # terms for the merged profile.
                    V_mat = G_mat @ U_mat @ G_mat.T

                    # TODO: use a chunk approach to speed things up.

                    # Most likely, I will have some non-diagonal elements to this matrix,
                    # indicative of correlations between the interpolated points (this will happen
                    # if a bin edge is used for two interpolated points, which is not that rare).
                    # I can't really do anything with this information for now, so I'll simply
                    # issue a warning.
                    # TODO: could one do better and account for this "somehow" ? This would most
                    # likely not be an easy update.
                    if not np.array_equal(V_mat[(V_mat != 0)|(np.isnan(V_mat))],
                                          V_mat.diagonal(), equal_nan=True):
                        logger.warning("Correlation between resampled profile elements " +
                                       " is being ignored. This is bad !")

                    # Assign the values to the new_data DataFrame
                    new_data.loc[:, name] = np.sqrt(V_mat.diagonal())

            # And finally let's assign the new DataFrame to the Profile. The underlying setter
            # will take care of reformatting all the indices as needed.
            prfs[prf_ind].data = new_data

            # Here, remember to still deal with flags. I'll mark anything that was interpolated.
            prfs[prf_ind].set_flg('interp', True,
                                  index=pd.Index([ind for (ind, val) in enumerate(omega_vals)
                                                  if val not in [0, 1]]))

            return prfs
