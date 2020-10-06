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
from ...database.database import db_mngr


# class FlagManager:
#     """Flag manager class"""
#
#     FLAG_BIT_NM = Flag.bit_number.name
#     FLAG_ABBR_NM = Flag.flag_abbr.name
#     FLAG_DESC_NM = Flag.flag_desc.name
#
#     def __init__(self, prfl_mngr):
#         """
#         Args:
#             prfl_mngr (ProfileManager): Profile manager
#         """
#         self._prfl_mngr = prfl_mngr
#         self._flags_abbr = {
#             arg[self.FLAG_ABBR_NM]: arg
#             for arg in db_mngr.get_flags()
#         }
#
#     @property
#     def prfl_mngr(self):
#         """ProfileManager: Profile manager"""
#         return self._prfl_mngr
#
#     @property
#     def flags_abbr(self):
#         """dict: Flag abbr, description and bit position."""
#         return self._flags_abbr
#
#     @property
#     def data(self):
#         """pandas.Series: Data corresponding flag value."""
#         return self.prfl_mngr.flag
#
#     @data.setter
#     def data(self, value):
#         self.prfl_mngr.flag = value
#
#     # def reindex(self, index, set_abbr):
#     #     """Reindex method
#     #
#     #     Args:
#     #         index (iterable): New index
#     #         set_abbr (str): Flag abbr used to replace NaN creating during
#     #             reindexing.
#     #
#     #     """
#     #
#     #     # Reindex flag
#     #     self.data = self.data.reindex(index)
#     #
#     #     # Set resampled flag
#     #     self.set_bit_val(set_abbr, True, self.data.isna())
#
#     def get_bit_number(self, abbr):
#         """Get bit number corresponding to given flag abbr"""
#         return self.flags_abbr[abbr][self.FLAG_BIT_NM]
#
#     def set_bit_val(self, abbr, set_val, index=None):
#         """Set data flag value to one
#
#         Args:
#             abbr (str):
#             set_val (bool): Default to True
#             index (pd.Index, optional): Default to None.
#
#         """
#
#         # Define
#         def set_to_true(x):
#             """Set bit to True"""
#             if np.isnan(x):
#                 out = (1 << self.get_bit_number(abbr))
#             else:
#                 out = int(x) | (1 << self.get_bit_number(abbr))
#
#             return out
#
#         def set_to_false(x):
#             """Set bit to False"""
#             if np.isnan(x):
#                 out = 0
#             else:
#                 out = int(x) & ~(1 << self.get_bit_number(abbr))
#
#             return out
#
#         # Init
#         if index is None:
#             index = self.data.index
#
#         # Set bit
#         if set_val is True:
#             self.flag.loc[index] = self.data.loc[index].apply(set_to_true)
#         else:
#             self.data.loc[index] = self.data.loc[index].apply(set_to_false)
#
#     def get_bit_val(self, abbr):
#         """Get data flag value
#
#         Args:
#             abbr (str):
#
#         """
#         bit_nbr = self.get_bit_number(abbr)
#         return self.data.apply(lambda x: (x >> bit_nbr) & 1)


class ProfileManager(metaclass=ABCMeta):
    """Profile manager"""

    FLAG_BIT_NM = Flag.bit_number.name
    FLAG_ABBR_NM = Flag.flag_abbr.name
    FLAG_DESC_NM = Flag.flag_desc.name

    def __init__(self, event_mngr, value=None, flag=None):
        """Constructor

        Args:
            event_mngr (int): Event id
            value:
            flag:

        """

        # Set attributes
        self._data = pd.DataFrame(columns=['value', 'flag'], dtype=float)
        self._event_mngr = event_mngr
        self._flags_abbr = {
            arg[self.FLAG_ABBR_NM]: arg
            for arg in db_mngr.get_flags()
        }

        # Set value and flag
        if value is not None:
            self.value = value
        if flag is not None:
            self.flag = flag

    @property
    def data(self):
        """pd.DataFrame: Data"""
        return self._data

    # @data.setter
    # def data(self, val):
    #
    #     # Test index
    #     self._test_index(val.index)
    #
    #     if len(self) == 0:
    #         self._data = val
    #     else:
    #         self.data.update(val)

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
        """pd.Series: Corresponding data value"""
        return self.data['value']

    @value.setter
    def value(self, val):

        # Test arg
        self._test_index(val.index)

        # Set series name
        val.name = 'value'

        if len(self) == 0:
            self.data['value'] = val
        else:

            self.data.update(val)

    @property
    def flag(self):
        """pd.Series: Corresponding data flag"""
        return self.data['flag']

    @flag.setter
    def flag(self, val):

        # Test arg
        self._test_index(val.index)

        # Set series name
        val.name = 'flag'

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

    def interpolate(self):
        """Interpolated method"""

        #TODO
        # Add automatic interpolation for polar coord (e.g. wind direction)
        # Check if this function must be improved/fixed
        interp_data = self.data.interpolate(method='index')

        # Set interp flag
        self.flag_mngr.set_bit_val('interp', True, index=self.data.isna())

        # Set data
        self.data = interp_data

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
            raise IndexError('Bad index type')

#
#     @staticmethod
#     def factory(data, event_mngr, index_lag=pd.Timedelta('0s'), u_u=None, u_r=None, u_s=None, u_t=None):
#         """TimeProfileManager factory"""
#
#         if len(data) == 0:
#             return EmptyProfileManager(event_mngr)
#         if (u_r is None) or (u_s is None) or (u_t is None):
#             return TimeProfileManager(data, event_mngr, index_lag)
#         if (u_s is None) or (u_t is None):
#             raise NotImplementedError()
#             #return TimeProfileErrTypeA(data, event_mngr, index_lag, err_r)
#         else:
#             raise NotImplementedError()
#             #return TimeProfileErrTypeA(data, event_mngr, index_lag, err_r, err_s, err_t)
#
#     def __init__(self, data, event_mngr, index_lag):
#         """Constructor
#
#         Args:
#             data (pd.Series): pd.Series with index of type pd.TimedeltaIndex
#             event_mngr (EventManager):
#             index_lag (pd.Timedelta):
#
#         """
#         super().__init__(data, event_mngr)
#
#         # Test
#         assert isinstance(index_lag, pd.Timedelta)
#
#         # Init attributes
#         self._index_lag = index_lag
#
#         # Reset index
#         self._reset_index()
#
#         # Set raw NA
#         self._flag_mngr.set_bit_val('raw_na', True, self.data.isna())
#
#     @property
#     def index_lag(self):
#         """pd.Timedelta: Index time lag"""
#         return self._index_lag
#
#     def check_data(self, val):
#         """Overwrite check data"""
#         super().check_data(val)
#
#         # Test
#         assert isinstance(val.index, pd.TimedeltaIndex)
#
#
#     def _reset_index(self):
#         """Set index start at 0s"""
#         self._data.index = self.data.index - self.data.index[0]
#         self._flag_mngr.data.index = self.flag.index - self.flag.index[0]
#
#     def shift(self, periods):
#         """
#
#         Args:
#           periods:
#
#         Returns:
#
#
#         """
#
#         self._data = self.data.shift(periods)
#         self._index_lag -= pd.Timedelta(periods, self.data.index.freq.name)


class IndexError(Exception):
    """Index error exception"""
