"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data manager classes used in dvas.data.data.ProfileManager

"""

# Import from external packages
from copy import deepcopy
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd

# Import from current package
from ...database.model import Flag
from ...database.database import DatabaseManager


class ProfileManager(metaclass=ABCMeta):
    """Profile manager"""

    FLAG_BIT_NM = Flag.bit_number.name
    FLAG_ABBR_NM = Flag.flag_abbr.name
    FLAG_DESC_NM = Flag.flag_desc.name

    db_mngr = DatabaseManager()

    def __init__(self, event_mngr, data=None):
        """Constructor

        Args:
            event_mngr (int): Event id
            data:

        """

        # Set attributes
        self._data = pd.concat(
                [pd.Series(dtype='float'), pd.Series(dtype='Int64')],
                axis=1
            )
        self._data.columns = ['value', 'flag']
        self._event_mngr = event_mngr
        self._flags_abbr = {
            arg[self.FLAG_ABBR_NM]: arg
            for arg in self.db_mngr.get_flags()
        }

        # Set data
        if data is not None:
            self.data = data

    @property
    def data(self):
        """pd.DataFrame: Data. Columns index, 0: value, 1: flag"""
        return self._data

    @data.setter
    def data(self, val):

        # Test arg
        self._test_index(val.index)

        # Set
        self.value = val.iloc[:, 0]
        self.flag = val.iloc[:, 1]

    @property
    def event_mngr(self):
        """EventManager: Corresponding data event manager"""
        return self._event_mngr

    @property
    def flags_abbr(self):
        """dict: Flag abbr, description and bit position."""
        return self._flags_abbr

    @property
    def value(self):
        """pd.Series: Corresponding data 'value'"""
        return self.data['value']

    @value.setter
    def value(self, val):

        # Test arg
        self._test_index(val.index)

        # Set series name/dtype
        val.name = 'value'
        val = val.astype('float')

        if len(self) == 0:
            self.data['value'] = val
        else:
            self.data.update(val)

    @property
    def flag(self):
        """pd.Series: Corresponding data 'flag'"""
        return self.data['flag']

    @flag.setter
    def flag(self, val):

        # Test arg
        self._test_index(val.index)

        # Set series name/dtype (Int64 uses special NaN for integers)
        val.name = 'flag'
        val = val.astype('Int64')

        if len(self) == 0:
            self.data['flag'] = val
        else:
            self.data.update(val)

    def copy(self):
        """Copy method"""
        return deepcopy(self)

    def __len__(self):
        return len(self.data)

    def _get_flag_bit_nbr(self, abbr):
        """Get bit number corresponding to given flag abbr"""
        return self.flags_abbr[abbr][self.FLAG_BIT_NM]

    def set_flag_val(self, abbr, set_val, index=None):
        """Set data flag value to one

        Args:
            abbr (str):
            set_val (bool): Default to True
            index (pd.Index, optional): Default to None.

        """

        # Define
        def set_to_true(x):
            """Set bit to True"""
            if np.isnan(x):
                out = (1 << self._get_flag_bit_nbr(abbr))
            else:
                out = int(x) | (1 << self._get_flag_bit_nbr(abbr))

            return out

        def set_to_false(x):
            """Set bit to False"""
            if np.isnan(x):
                out = 0
            else:
                out = int(x) & ~(1 << self._get_flag_bit_nbr(abbr))

            return out

        # Init
        if index is None:
            index = self.data.index

        # Set bit
        if set_val is True:
            self.flag = (index, self.flag.loc[index].apply(set_to_true))
        else:
            self.flag = (index, self.flag.loc[index].apply(set_to_false))

    def get_flag_val(self, abbr):
        """Get data flag value

        Args:
            abbr (str):

        """
        bit_nbr = self._get_flag_bit_nbr(abbr)
        return self.flag.apply(lambda x: (x >> bit_nbr) & 1)

    @abstractmethod
    def _test_index(self, index):
        """Test index

        Args:
            index (pd.Index): Data index

        Raises:
            IndexError: Bad index type
        """


class TimeProfileManager(ProfileManager):
    """Time profile manager """

    def _test_index(self, index):
        """Implementation"""

        try:
            assert isinstance(index, pd.TimedeltaIndex)

        except AssertionError:
            raise IndexError(
                'Bad index type. Must be a pd.TimedeltaIndex class'
            )
