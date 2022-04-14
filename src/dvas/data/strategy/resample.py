"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Resample strategies

"""

# Import from external packages
import logging
import functools
import multiprocessing as mp
import numpy as np
import pandas as pd

# Import from current package
from ...errors import DvasError
from .data import MPStrategyAC
from ...tools.gdps.utils import process_chunk
from ...tools.tools import df_to_chunks
from ...hardcoded import PRF_REF_INDEX_NAME, PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_FLG_NAME
from ...hardcoded import PRF_REF_VAL_NAME, PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME
from ...hardcoded import PRF_REF_UCU_NAME

# Steup the logger
logger = logging.getLogger(__name__)


class ResampleStrategy(MPStrategyAC):
    """ Class to handle the resample strategy for RS and GDPs Profiles. """

    def execute(self, prfs, freq: str = '1s', chunk_size: int = 150, n_cpus: int = 1):
        """Implementation of time resampling method for RS and GDP Profiles.

        .. note::

            This strategy does NOT treat NaN's in a special way. This implies that if a NaN is one
            of the two closest original data points from a new location to be interpolated, that
            location will result in a NaN as well.

        Args:
            prfs (list of RSProfiles|GDPProfiles): GDP Profiles to resample.
            freq (str, optional): see pandas.timedelta_range(). Defaults to '1s'.
            chunk_size (int, optional): to speed up computation, Profiles get broken up in chunks of
                that length. The larger the chunks, the larger the memory requirements. The smaller
                the chunks the more items to process. Defaults to 150.
            n_cpus (int|str, optional): number of cpus to use. Can be a number, or 'max'. Set to 1
                to disable multiprocessing. Defaults to 1.

        Returns:
            dvas.data.MultiRSProfile|MultiGDPProfile: the resampled MultiProfile.

        """

        # Some sanity checks to begin with
        if not isinstance(prfs, list):
            raise DvasError("Ouch ! prfs should be of type list, and not: {}".format(type(prfs)))
        # The following should in principle never happen because the strategy ensures that.
        # If this blows up, then something must have gone really wrong ...
        if np.any([PRF_REF_TDT_NAME not in prf.get_index_attr() for prf in prfs]):
            raise DvasError("Ouch ! I can only resample profiles with a timedelta array ...")

        # Very well, let's start looping and resampling each Profile
        for (prf_ind, prf) in enumerate(prfs):

            # Here, do something only if I actually have data to resample
            if len(prf.data) <= 1:
                continue

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
                    logger.info('No resampling required for %s', prfs[prf_ind].info.src)
                    continue

            logger.warning('Starting resampling for %s', prfs[prf_ind].info.src)
            # Very well, interpolation is required. To avoid duplicating code, we shall rely on
            # the dvas.tools.gdps.utils.process_chunk() function to do so.
            # This implies that we must construct a suitable set of df_chunks to feed that function.
            # These chunks must be comprised of two "profiles", that when *subtracted* with a
            # binning=1 will give me the interpolated profile.
            # This two profiles must thus already include any weight necessary to the interpolation

            # Let's begin by creating the chunk array
            cols = [[0, 1],
                    [PRF_REF_TDT_NAME, PRF_REF_ALT_NAME, PRF_REF_VAL_NAME, PRF_REF_FLG_NAME,
                     PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME,
                     'uc_tot', 'oid', 'mid', 'eid', 'rid']]
            cols = pd.MultiIndex.from_product(cols)
            x_dx = pd.DataFrame(columns=cols)
            x_dx.index.name = PRF_REF_INDEX_NAME

            # Next, get the original data out, reseting all the indices to columns.
            this_data = prf.data.reset_index()

            # Let's drop the original integer index, to avoid type conversion issues
            this_data.drop(columns=PRF_REF_INDEX_NAME, inplace=True)

            # Duplicate the last point as a "pseudo" new time step.
            # This is to ensure proper interpolation all the way to the very edge of the raw data
            this_data = pd.concat([this_data, this_data.iloc[-1:]], axis=0, ignore_index=True)
            this_data.iloc[-1, this_data.columns.get_loc('tdt')] += pd.Timedelta(1, 's')
            # Re-extract the old_tdt with this extra row
            old_tdt = this_data['tdt'].values

            # What are the indices of the closest (upper) points for each (new) step ?
            x_ip1_ind = np.array([this_data[PRF_REF_TDT_NAME].searchsorted(xi_star,
                                  side='right') for xi_star in new_tdt])

            # None of these should be smaller than 0 or larger than the length of the original array
            # And they should all be real.
            assert all((x_ip1_ind >= 0) * (x_ip1_ind < len(this_data)))

            # What are the linear interpolation weights ?
            # Here, we have x_- * (1-omega) + x_+ * (omega)
            omega_vals = np.array([(item-old_tdt[x_ip1_ind[ind]-1]) /
                                   np.diff(old_tdt)[x_ip1_ind[ind]-1]
                                   for (ind, item) in enumerate(new_tdt.values)])

            # All these weights should be comprised between 0 and 1 ... else something is reall bad.
            assert all((omega_vals >= 0) * (omega_vals <= 1))

            # I am now ready to "fill the chunks". The first profile will be
            # x_- * (omega-1) and the second will be x_+ * omega. That way, 1-2 =
            # x_- * (1-omega) + x_+ * omega.
            for col in this_data.columns:
                if col == PRF_REF_FLG_NAME:  # Do nothing to the flags
                    x_dx.loc[:, (0, col)] = this_data.iloc[x_ip1_ind-1][col].values
                    x_dx.loc[:, (1, col)] = this_data.iloc[x_ip1_ind][col].values
                else:
                    # Here note that we multiply by (omega-1) instead of omega
                    # This is so that we can "disguise" the combination of profiles as a delta
                    # (rather than a sum) for compatibility with process_chunk()
                    x_dx.loc[:, (0, col)] = this_data.iloc[x_ip1_ind-1][col].values * (omega_vals-1)
                    x_dx.loc[:, (1, col)] = this_data.iloc[x_ip1_ind][col].values * omega_vals

            # Deal with the uncertainties, in case I do not have a GDPProfile
            for col in [PRF_REF_UCR_NAME, PRF_REF_UCS_NAME, PRF_REF_UCT_NAME, PRF_REF_UCU_NAME]:
                if col not in this_data.columns:
                    x_dx.loc[:, (0, col)] = 0
                    x_dx.loc[:, (1, col)] = 0

            # Also deal with the total uncertainty
            try:
                x_dx.loc[:, (0, 'uc_tot')] = prf.uc_tot.iloc[x_ip1_ind-1].values
                x_dx.loc[:, (1, 'uc_tot')] = prf.uc_tot.iloc[x_ip1_ind].values
            except AttributeError:
                x_dx.loc[:, (0, 'uc_tot')] = 0
                x_dx.loc[:, (1, 'uc_tot')] = 0

            # Assign the oid, eid, mid, rid values. Since we are here resampling one profile,
            # they are the same for all (and thus their value is irrelevant)
            for col in ['eid', 'rid', 'mid']:
                x_dx.loc[:, (0, col)] = 0
                x_dx.loc[:, (1, col)] = 0

            # WARNING: here, we set the oid to be different for the two "profiles".
            # This is not correct, strictly speaking, since all the data comes from the "same"
            # profile that we interpolate. However, if we do not do that, I have currently
            # no way to adjust the index in the gdps.utils.correlation() function,
            # and thus the errors are badly calculated. Since the 'oid' is not yet used for anything
            # in the correlation matrix, we can use this as an "alternative" to say that two
            # points with the same index are different. But the day that the oid is being used,
            # then this will blow up. Badly.
            x_dx.loc[:, (0, 'oid')] = 0
            x_dx.loc[:, (1, 'oid')] = 1

            # Break this into chunks to speed up calculation
            # WARNING: This is possible only by assuming that there is no cross-correlation between
            # neighboring values, which is not strictly true, since a given value could be used
            # as an interpolation  node for multiple steps. Here, we're foolishly ignoring this
            # real possibility entirely.
            chunks = df_to_chunks(x_dx, chunk_size)

            # Prepare a routine to dispatch chunks to multiple cpus
            merge_func = functools.partial(process_chunk, binning=1, method='delta')
            if n_cpus == 1:
                proc_chunks = map(merge_func, chunks)
            else:

                # TODO: this is a bug that I do not understand. When running the script from an
                # ipython session multiple times in a row, the __spec__ variable is only defined
                # the first time around. Having this set to anything (=None when run interactively)
                # is crucial to the multiprocessing Pool routine.
                # So for now, use a terribly dangerous workaround that I do not understand.
                # This should definitely be fixed. Or not ?
                # See #121
                import sys
                try:
                    sys.modules['__main__'].__spec__ is None
                except AttributeError:
                    logger.warning('BUG: __spec__ is not set. Fixing it by hand ...')
                    sys.modules['__main__'].__spec__ = None

                pool = mp.Pool(processes=n_cpus)
                proc_chunks = pool.map(merge_func, chunks)
                pool.close()
                pool.join()

            # Re-assemble all the chunks into one DataFrame.
            proc_chunk = pd.concat(proc_chunks, axis=0)

            # Start the interpolation. Since I already applied the weights, I just need to do a
            # delta.
            # out = process_chunk(x_dx, binning=1, method='delta')

            # Let's make sure I have the correct times ... just to make sure nothing got messed up.
            assert (proc_chunk.tdt.round('s') == new_tdt).all()

            # Create a new dataframe to keep the interpolated stuff
            new_data = pd.DataFrame(new_tdt, columns=this_data.columns)

            # Loop through the different columns to assign
            for name in this_data.columns:
                if name == PRF_REF_TDT_NAME:
                    # Here, do nothing to avoid propagating floating point errors caused by the
                    # interpolation of time steps.
                    continue

                if name == PRF_REF_FLG_NAME:
                    # For the points that were not interpolated, copy them over
                    # For the others, do nothing as we will set them properly below
                    # Treat the case of omega_vals =0/1 differently, to assign the corret flags
                    new_data.loc[np.flatnonzero(omega_vals == 0), name] = \
                        this_data.loc[x_ip1_ind[omega_vals == 0] - 1, name].values

                    new_data.loc[np.flatnonzero(omega_vals == 1), name] = \
                        this_data.loc[x_ip1_ind[omega_vals == 1], name].values

                    continue

                # For all the rest, let's just use the data that was recently computed
                new_data.loc[:, name] = proc_chunk.loc[:, name].values

            # And finally let's assign the new DataFrame to the Profile. The underlying setter
            # will take care of reformatting all the indices as needed.
            prfs[prf_ind].data = new_data

            # Here, remember to still deal with flags. I'll mark anything that was interpolated.
            prfs[prf_ind].set_flg('interp', True,
                                  index=pd.Index([ind for (ind, val) in enumerate(omega_vals)
                                                  if val not in [0, 1]]))

        return prfs
