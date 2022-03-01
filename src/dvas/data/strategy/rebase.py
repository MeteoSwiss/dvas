"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Rebase strategy

"""

# Import from external packages
import numbers
import pandas as pd

# Import from current package
from .data import MPStrategyAC
from ...errors import DvasError
from ...hardcoded import PRF_REF_INDEX_NAME


class RebaseStrategy(MPStrategyAC):
    """Class to manage time data synchronization"""

    def execute(self, prfs, new_lengths, shifts=None):
        """ Rebases Profiles on a DataFrame with a different length, possibly shifting values
        around.

        Any missing data gets filled with NaN/NaT. Any superfulous data is be cropped.
        All non-integer indices get rebased as well (i.e. they are NOT interpolated).

        Args:
            prfs (list of dvas.data.strategy.data.Profile|RSProfile|GDPProfile): Profiles to
                rebase.
            new_lengths (int|list of int): The new lengths for the Profiles. Will crop/add empty
                elements at the *end* of the profiles as needed.
            shifts (int|list of int, optional): row n of the existing data will become row n+shift.
                If specifiying an int, the same shift will be applied to all Profiles. Else, the
                list should specify a shift for each Profile. Defaults to None.

        Returns:
            dvas.data.MultiProfile|MultiRSProfile|MultiGDPProfile: the rebased MultiProfile.

        """

        # Make sure I get fed a list of profiles.
        if not isinstance(prfs, list):
            raise DvasError("Ouch ! prfs should be of type list, and not: {}".format(type(prfs)))

        # Deal with the length(s). Try to be as courteous as possible.
        if isinstance(new_lengths, numbers.Integral):
            new_lengths = [new_lengths] * len(prfs)
        elif isinstance(new_lengths, list):
            if not all([isinstance(item, numbers.Integral) for item in new_lengths]):
                raise DvasError("Ouch ! new_lengths should be a list of int.")
            if len(new_lengths) != len(prfs):
                if len(set(new_lengths)) == 1:
                    new_lengths = [new_lengths[0]] * len(prfs)
                else:
                    raise DvasError("Ouch ! new_lengths should have length of %i, not %i" %
                                    (len(prfs), len(new_lengths)))
        else:
            raise DvasError("Type %s unspported for new_length. I need int|list of int." %
                            type(new_lengths))

        # Deal with the shift(s). Still try to be as courteous as possible.
        if shifts is None:
            shifts = [0] * len(prfs)
        if isinstance(shifts, numbers.Integral):
            shifts = [shifts] * len(prfs)
        elif isinstance(shifts, list):
            if not all([isinstance(item, numbers.Integral) for item in shifts]):
                raise DvasError("Ouch ! shift should be a list of int.")
            if len(shifts) != len(prfs):
                if len(set(shifts)) == 1:
                    shifts = [shifts[0]] * len(prfs)
                else:
                    raise DvasError("Ouch ! shifts should have length of %i, not %i" %
                                    (len(prfs), len(shifts)))
        else:
            raise DvasError("Type %s unspported for shifts. I need int|list of int." % type(shifts))

        # We are good to go: let's loop through the different Profiles.
        for (prf_ind, prf) in enumerate(prfs):

            # Get the data out, reseting all the indices, because they are a pain to deal with.
            this_data = prf.data.reset_index()

            # Let's also drop the original integer index, to avoid type conversion issues
            this_data.drop(columns=PRF_REF_INDEX_NAME, inplace=True)

            # Shift the index of the rows as required
            this_data.index += shifts[prf_ind]

            # Create the new data, full of NaNs but with all the suitable columns.
            new_data = pd.DataFrame(index=range(new_lengths[prf_ind]), columns=this_data.columns)

            # Here let's make sure this new data has the proper column types.
            # Do it one by one, because I miserably failed at figuring out something more elegant.
            for col in new_data.columns.to_list():
                new_data[col] = new_data[col].astype(this_data[col].dtype)

            # Fill the new data where needed. Note here that this line relies on the fact that
            # pandas will take care of replacing only the rows that have the same indices.
            # Anything that needs to be dropped from this_data will thus be dropped as required.
            new_data.update(this_data)

            # And finally let's assign the new DataFrame to the Profile. The underlying setter
            # will take care of reformatting all the indices as needed.
            prfs[prf_ind].data = new_data

        return prfs
