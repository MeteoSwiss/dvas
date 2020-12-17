"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Synchronizer strategy

"""

# Import from external packages
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd

# Import from current package
from ...errors import dvasError
from ...logger import data as logger

class AbstractRebaseStrategy(metaclass=ABCMeta):
    """Abstract class to manage data rebasing (=index change) strategy"""

    @ abstractmethod
    def rebase(self, *args, **kwargs):
        """Strategy required method"""


class RebaseStrategy(AbstractRebaseStrategy):
    """Class to manage time data synchronization"""

    def rebase(self, multiprf, new_index, shift=None):
        """ Rebase MultiProfiles on new indices.

        Any missing data will be filled with NaN. Any superflous data will be cropped.

        Args:
            multiprf (dvas.data.MultiProfile|MultiRSProfile|MultiGDPProfile): MultiProfile to
                rebase.
            new_index (pandas.core.indexes.multi.MultiIndex): The new indices to rebase upon.
            shift (int|list of int, optional): row n of the existing data will become row n+shift.
                If specifiying an int, the same shift will be applied to all Profiles. Else, the
                list of should specify a shift for each Profile. Defaults to None.

        Returns:
            dvas.data.MultiProfile|MultiRSProfile|MultiGDPProfile: the rebased MultiProfile.

        """

        # Deal with the shifts. Try to be as courteous as possible.
        if shift is None:
            shift = [0] * len(multiprf)
        if isinstance(shift, int):
            shift = [shift] * len(multiprf)
        elif isinstance(shift, list):
            if not all([isinstance(item, int) for item in shift]):
                raise dvasError("Ouch ! shift should be a list of int.")
            if len(shift) != len(multiprf):
                if len(set(shift)) == 1:
                    shift = shift * len(multiprf)
                else:
                    raise dvasError("Ouch ! shift should have length of %i, not %i" %
                                    (len(multiprf), len(shift)))
        else:
            raise dvasError("Type %s unspported for shift. I need str|list of str." % type(shift))

        # First, get the profiles out
        prfs = multiprf.profiles

        # First, let us ensures that I have the same kind of indices
        if not np.all([new_index.names == prf.data.index.names for prf in prfs]):
            logger.warning('Rebasing profile on inconsistent indices: %s', new_index.names)

        # We are good to go: let's loop through the different Profiles.
        for (prf_ind, prf) in enumerate(prfs):

            # Create the new data, full of NaNs but with all the suitable columns.
            new_data = pd.DataFrame(index=new_index, columns=prf.data.columns)

            # Here let's make sure this new data has the proper column types.
            # Do it one by one, because I miserably failed at figuring out something more elegant.
            for col in new_data.columns.to_list():
                new_data[col] = new_data[col].astype(prf.data[col].dtype)

            # Let's figure out which part of the data needs to go where.
            new_start = np.max([0, shift[prf_ind]])
            new_end = np.min([len(new_index), len(prf.data) + shift[prf_ind]])
            old_start = np.max([0, -shift[prf_ind]])
            old_end = old_start + (new_end-new_start)

            # Make a quick check that I did not mess things up
            if new_end - new_start != old_end-old_start:
                raise dvasError("This error is impossible !")

            # Let's rebase the data. Pay close attention to the fact that since the old and new
            # DataFrame are likely to NOT have the same indices, I must got through numpy to copy
            # the values over. Sigh....
            new_data[new_start:new_end] = prf.data[old_start:old_end].to_numpy()

            # And finally let's assign it
            multiprf.profiles[prf_ind].data = new_data

        return multiprf
