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
from ...tools.chunks import process_chunk
from ...tools.tools import df_to_chunks, fancy_bitwise_or
from ...hardcoded import PRF_IDX, PRF_TDT, PRF_ALT, PRF_VAL, PRF_UCS, PRF_UCT, PRF_UCU
from ...hardcoded import PRF_FLG, FLG_INTERP, TAG_1S

# Setup the logger
logger = logging.getLogger(__name__)


class ResampleStrategy(MPStrategyAC):
    """ Class to handle the resample strategy for RS and GDPs Profiles. """

    def execute(self, prfs, freq: str = '1s', chunk_size: int = 150, n_cpus: int = 1,
                interp_dist: float | int = 1, circular: bool = False):
        """ Implementation of time resampling method for RS and GDP Profiles.

        Args:
            prfs (list of RSProfiles|GDPProfiles): GDP Profiles to resample.
            freq (str, optional): see pandas.timedelta_range(). Defaults to '1s'.
            interp_dist (float|int, optional): define the distance between resampled points and
                their closest measurement, in s, from which the resampled element is forced to NaN.
                Defaults to 1, i.e. gaps that are >= 1s are not interpolated.
            chunk_size (int, optional): to speed up computation, Profiles get broken up in chunks of
                that length. The larger the chunks, the larger the memory requirements. The smaller
                the chunks the more items to process. Defaults to 150.
            n_cpus (int|str, optional): number of cpus to use. Can be a number, or 'max'. Set to 1
                to disable multiprocessing. Defaults to 1.
            circular (bool, optional): if True, will assume angular values and use np.unwrap()
                before interpolating. Defaults to False.

        Returns:
            dvas.data.MultiRSProfile|MultiGDPProfile: the resampled MultiProfile.

        .. note::

            This strategy does NOT treat NaN's in a special way. This implies that if a NaN is one
            of the two closest original data points from a new location to be interpolated, that
            location will result in a NaN as well.

        """

        # Some sanity checks to begin with
        if not isinstance(prfs, list):
            raise DvasError(f"prfs should be of type list, and not: {type(prfs)}")
        # The following should in principle never happen because the strategy ensures that.
        # If this blows up, then something must have gone really wrong ...
        if np.any([PRF_TDT not in prf.get_index_attr() for prf in prfs]):
            raise DvasError("I can only resample profiles with a timedelta array ...")

        # Very well, let's start looping and resampling each Profile
        for (prf_ind, prf) in enumerate(prfs):

            # Here, do something only if I actually have data to resample
            if len(prf.data) <= 1:
                continue

            logger.info('Checking that timesteps are increasing monotically ... (%s)',
                        prf.info.src)
            is_bad = True
            while is_bad:
                # Compute the time deltas
                # WARNING: we here use the apply method and a lambda function,
                # to avoid floating point errors related to
                # https://github.com/pandas-dev/pandas/issues/34290

                tsteps = pd.Series(prf.data.index.get_level_values(PRF_TDT)).apply(
                                   lambda x: x.total_seconds())

                if any(bad := (tsteps.diff() < 0)):

                    logger.warning('Found %i decreasing timesteps. Cropping them now. (%s)',
                                   len(bad[bad]), prf.info.src)

                elif any(bad := (tsteps.diff() == 0)):
                    logger.warning('Found %i duplicated timesteps. Cropping them now. (%s)',
                                   len(bad[bad]), prf.info.src)
                else:
                    is_bad = False

                # If applicable, crop the bad points
                if is_bad:
                    prf.data = prf.data[~bad.values]
                    # Sanity check that the IDX index remains ok.
                    assert all(np.diff(prf.data.index.get_level_values('_idx')) == 1)

            # Let's identify the min and max integer values, rounded to the nearest second.
            t_0 = min(prf.data.index.get_level_values(PRF_TDT)).ceil('1s')
            t_1 = max(prf.data.index.get_level_values(PRF_TDT)).floor('1s')

            # Turn this into a regular grid
            new_tdt = pd.timedelta_range(t_0, t_1, freq=freq, name=PRF_TDT)

            # Get the existing time steps
            old_tdt = prf.data.index.get_level_values(PRF_TDT)

            # Check if the new index is the same as the old. And if so, continue to the next
            # next profile without changing anything
            if len(new_tdt) == len(prf.data):
                if all(new_tdt == prf.data.index.get_level_values(PRF_TDT)):
                    logger.info('No resampling required for %s', prf.info.src)
                    continue

                logger.warning('Non-integer time steps (%s).', prfs[prf_ind].info.src)

            elif len(new_tdt) < len(prf.data):
                logger.warning('Extra-numerous timesteps (%s).', prfs[prf_ind].info.src)
            else:
                logger.warning('Missing (at least) %i time steps (%s).',
                               len(new_tdt) - len(prf.data), prf.info.src)

            # dvas should never resample anything. If we do, let's make it very visible.
            logger.warning('Starting resampling (%s)', prf.info.src)
            # Very well, interpolation is required. To avoid duplicating code, we shall rely on
            # the dvas.tools.gdps.utils.process_chunk() function to do so.
            # This implies that we must construct a suitable set of df_chunks to feed that function.
            # These chunks must be comprised of two "profiles", that when *subtracted* with a
            # binning=1 will give me the interpolated profile.
            # This two profiles must thus already include any weight necessary to the interpolation

            # Let's begin by creating the chunk array
            cols = [[0, 1],
                    [PRF_TDT, PRF_ALT, PRF_VAL, PRF_FLG, PRF_UCS, PRF_UCT, PRF_UCU,
                     'uc_tot', 'oid', 'mid', 'eid', 'rid']]
            cols = pd.MultiIndex.from_product(cols)
            x_dx = pd.DataFrame(columns=cols)
            x_dx.index.name = PRF_IDX

            # Next, get the original data out, reseting all the indices to columns.
            this_data = prf.data.reset_index()

            # Let's drop the original integer index, to avoid type conversion issues
            this_data.drop(columns=PRF_IDX, inplace=True)

            # Unwrap angles if necessary
            if circular:
                # Fix 273: ignore NaNs in unwrap, following the suggestion by ecatmur on SO
                # https://stackoverflow.com/questions/37027295
                valids = ~this_data[PRF_VAL].isna().values
                this_data.loc[valids, PRF_VAL] = \
                    np.rad2deg(np.unwrap(np.deg2rad(this_data.loc[valids, PRF_VAL].values)))

            # Duplicate the last point as a "pseudo" new time step.
            # This is to ensure proper interpolation all the way to the very edge of the original
            # data
            this_data = pd.concat([this_data, this_data.iloc[-1:]], axis=0, ignore_index=True)
            this_data.iloc[-1, this_data.columns.get_loc('tdt')] += pd.Timedelta(1, 's')
            # Re-extract the old_tdt with this extra row
            old_tdt = this_data['tdt'].values

            # What are the indices of the closest (upper) points for each (new) step ?
            x_ip1_ind = np.array([this_data[PRF_TDT].searchsorted(xi_star,
                                  side='right') for xi_star in new_tdt])

            # None of these should be smaller than 0 or larger than the length of the original array
            # And they should all be real.
            assert all((x_ip1_ind >= 0) * (x_ip1_ind < len(this_data)))

            # What are the linear interpolation weights ?
            # Here, we have x_- * (1-omega) + x_+ * (omega)
            # Note: for circular resampling (with angluar values), as long as we have only two
            # angles, the arithmetic average is equivalent to the circular one. So we worry not.
            omega_vals = np.array([(item-old_tdt[x_ip1_ind[ind]-1]) /
                                   np.diff(old_tdt)[x_ip1_ind[ind]-1]
                                   for (ind, item) in enumerate(new_tdt.values)])

            # All these weights should be comprised between 0 and 1 ... else something went bad.
            assert all((omega_vals >= 0) * (omega_vals <= 1))

            # If the gap is large, the weights should be NaNs. We want to resample, NOT interpolate.
            # Let's find any point that is 1s or more away from a real measurement, and block these.
            to_hide = [np.min(np.abs(this_data['tdt'].dt.total_seconds().values - item))
                       for item in new_tdt.total_seconds().values]  # noqa pylint: disable=no-member
            to_hide = np.array(to_hide) >= interp_dist

            if any(to_hide):
                logger.warning('Resampling %i points to NaN (>=%.3fs from real data) (%s).',
                               len(to_hide[to_hide]), interp_dist, prf.info.src)
                omega_vals[to_hide] = np.nan

            # I am now ready to "fill the chunks". The first profile will be
            # x_- * (omega-1) and the second will be x_+ * omega. That way, 1-2 =
            # x_- * (1-omega) + x_+ * omega.
            for col in this_data.columns:
                if col == PRF_FLG:  # Do nothing to the flags
                    x_dx[(0, col)] = this_data.iloc[x_ip1_ind-1][col].values
                    x_dx[(1, col)] = this_data.iloc[x_ip1_ind][col].values
                else:
                    # Here note that we multiply by (omega-1) instead of omega
                    # This is so that we can "disguise" the combination of profiles as a delta
                    # (rather than a sum) for compatibility with process_chunk()
                    x_dx[(0, col)] = this_data.iloc[x_ip1_ind-1][col].values * (omega_vals-1)
                    x_dx[(1, col)] = this_data.iloc[x_ip1_ind][col].values * omega_vals

                    # When I multiply by 0, make sure the NaNs disappear. Else I risk propagating
                    # it while it is not required. What follows may seem convoluted, but it should
                    # ensure that the column types do not get messed up in the process.
                    x_dx.loc[omega_vals-1 == 0, (0, col)] = \
                        pd.Series([0]).astype(x_dx[(0, col)].dtype).values[0]
                    x_dx.loc[omega_vals == 0, (1, col)] = \
                        pd.Series([0]).astype(x_dx[(1, col)].dtype).values[0]

            # Deal with the uncertainties, in case I do not have a GDPProfile
            for col in [PRF_UCS, PRF_UCT, PRF_UCU]:
                if col not in this_data.columns:
                    # To avoid warnings down the line, set the UC to 0 everywhere, except where
                    # the value is a NaN.
                    x_dx[(0, col)] = [item if np.isnan(item) else 0 for item in omega_vals]
                    x_dx[(1, col)] = [item if np.isnan(item) else 0 for item in omega_vals]

            # Also deal with the total uncertainty
            try:
                x_dx[(0, 'uc_tot')] = prf.uc_tot.iloc[x_ip1_ind-1].values
                x_dx[(1, 'uc_tot')] = prf.uc_tot.iloc[x_ip1_ind].values
            except AttributeError:
                x_dx[(0, 'uc_tot')] = [item if np.isnan(item) else 0
                                       for item in x_dx.loc[:, (0, PRF_VAL)]]
                x_dx[(1, 'uc_tot')] = [item if np.isnan(item) else 0
                                       for item in x_dx.loc[:, (1, PRF_VAL)]]

            # Assign the oid, eid, mid, rid values. Since we are here resampling one profile,
            # they are the same for all (and thus their value is irrelevant)
            for col in ['eid', 'rid', 'mid']:
                x_dx[(0, col)] = 0
                x_dx[(1, col)] = 0

            # WARNING: here, we set the oid to be different for the two "profiles".
            # This is not correct, strictly speaking, since all the data comes from the "same"
            # profile that we interpolate. However, if we do not do that, I have currently
            # no way to adjust the index in the gdps.utils.correlation() function,
            # and thus the errors are badly calculated. Since the 'oid' is not yet used for anything
            # in the correlation matrix, we can use this as an "alternative" to say that two
            # points with the same index are different. But the day that the oid is being used,
            # then this will blow up. Badly.
            x_dx[(0, 'oid')] = 0
            x_dx[(1, 'oid')] = 1

            # Break this into chunks to speed up calculation
            # WARNING: This is possible only by assuming that there is no cross-correlation between
            # neighboring values, which is not strictly true, since a given value could be used
            # as an interpolation  node for multiple steps. Here, we're foolishly ignoring this
            # real possibility entirely.
            chunks = df_to_chunks(x_dx, chunk_size)

            # Prepare a routine to dispatch chunks to multiple cpus
            merge_func = functools.partial(process_chunk, binning=1, method='arithmetic delta')
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
            proc_chunk = [item[0] for item in proc_chunks]
            proc_chunk = pd.concat(proc_chunk, axis=0)

            # Let's make sure I have the correct times ... just to make sure nothing got messed up.
            # Remember that some of the times may be NaNs, in case of large gaps ...
            notnans = proc_chunk.tdt.notna()
            assert (proc_chunk.tdt.round('s')[notnans] == new_tdt[notnans]).all()

            # Create a new dataframe to keep the interpolated stuff
            new_data = pd.DataFrame(new_tdt, columns=this_data.columns)

            # Loop through the different columns to assign
            for name in this_data.columns:
                if name == PRF_TDT:
                    # Here, do nothing to avoid propagating floating point errors and NaNs
                    # caused by the interpolation of the time steps.
                    continue

                if name == PRF_FLG:

                    # Assemble a DataFrame of meaningful flags, i.e. ignore those of the points
                    # that are not used for the interpolation. Typically those with weight of 0 or 1
                    flg_low = this_data.loc[x_ip1_ind - 1, PRF_FLG].mask(omega_vals == 1, 0)
                    flg_hgh = this_data.loc[x_ip1_ind, PRF_FLG].mask(omega_vals == 0, 0)
                    meaningful_flgs = pd.concat(
                        [flg_low.reset_index(drop=True), flg_hgh.reset_index(drop=True)], axis=1)

                    assert len(meaningful_flgs) == len(new_data), "flgs size mismatch"

                    # Very well, I am now ready to combine and assign these
                    # Fixes #259
                    new_data[name] = fancy_bitwise_or(meaningful_flgs, axis=1)
                    continue

                # For all the rest, let's just use the data that was recently computed
                new_data[name] = proc_chunk.loc[:, name].values

            # In case of angles, let's remmeber to bring these back to the [0;360[ range
            if circular:
                new_data[PRF_VAL] = new_data[PRF_VAL] % 360

            # And finally let's assign the new DataFrame to the Profile. The underlying setter
            # will take care of reformatting all the indices as needed.
            prfs[prf_ind].data = new_data

            # Here, remember to still deal with flags. I'll mark anything that was interpolated.
            prfs[prf_ind].set_flg(FLG_INTERP, True,
                                  index=pd.Index([ind for (ind, val) in enumerate(omega_vals)
                                                  if val not in [0, 1]]))

            # Let's also tag the entire Profile so they are easy to spot from the outside
            # (since resampling is definitely NOT what we want to do ... )
            assert any(prfs[prf_ind].has_flg(FLG_INTERP)), "All this for nothing ?!"
            prfs[prf_ind].info.add_tags(TAG_1S)

        return prfs
