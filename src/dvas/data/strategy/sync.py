"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Synchronizer strategy

"""

# Import from external packages
from abc import ABCMeta, abstractmethod
import numpy as np

# Import from current package
from ..math import crosscorr


class SynchronizeStrategy(metaclass=ABCMeta):
    """Abstract class to manage data synchronization strategy"""

    @ abstractmethod
    def synchronize(self, *args, **kwargs):
        """Strategy required method"""


class TimeSynchronizeStrategy(SynchronizeStrategy):
    """Class to manage time data synchronization"""

    def synchronize(self, data, ref_key, i_start=0, n_corr=300, window=30, i_ref=0):
        """Synchronise time profiles

        Notes:
            Only sorted data values of the same length can be synchronized.

        Args:
            data (dict): Dict of list of ProfileManager
            ref_key (str): Dict key use as sync reference
            i_start (int): Start integer index. Default to 0.
            n_corr (int): Default to 300.
            window (int): Window size. Default to 30.
            i_ref (int): Reference TimeProfile index. Default to 0.

        Returns:


        """

        # Check data ordering
        keys = list(data.keys())
        event_mngrs = {
            key: [arg.event_mngr for arg in val] for key, val in data.items()
        }
        tst = all(
            [event_mngrs[key] == event_mngrs[keys[0]] for key in keys[1:]]
        )
        assert tst is True, 'Unsorted data or of different length'

        # Create values and datas
        values = {
            key: [arg.value for arg in val] for key, val in data.items()
        }
        datas = {
            key: [arg.data for arg in val] for key, val in data.items()
        }

        # Select reference data
        ref_data = values[ref_key][i_ref]

        # Find best corrcoef
        idx_offset = []
        window_values = range(-window, window)
        for test_data in values[ref_key]:
            corr_res = [
                crosscorr(
                    ref_data.iloc[i_start:i_start + n_corr],
                    test_data.iloc[i_start:i_start + n_corr],
                    lag=w)
                for w in window_values
            ]
            idx_offset.append(np.argmax(corr_res))

        # Apply shift to data
        for key, val in datas.items():
            for i, arg in enumerate(val):
                val[i] = arg.shift(window_values[idx_offset[i]])

            data.update({key: val})

        return data
