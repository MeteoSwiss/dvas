"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data manager classes used in dvas.data.data.ProfileManager

"""

# Import from external packages
from abc import abstractmethod
import numpy as np
import pandas as pd

# Import from current package
from ...database.model import Flag
from ...database.database import DatabaseManager
from ...dvas_logger import dvasError
from ...dvas_helper import RequiredAttrMetaClass

# Define
INT_TEST = (np.int64, np.int, int, type(pd.NA))
FLOAT_TEST = (np.float, float) + INT_TEST
TIME_TEST = FLOAT_TEST + (pd.Timedelta,)


class ProfileAbstract(metaclass=RequiredAttrMetaClass):
    """Abstract Profile class"""

    # Specify required attributes
    # DF_COLS_ATTR dict items:
    # - 'test': types to test (tuple of type)
    # - 'type': type conversion (type|None)
    # - 'index': use as index (bool)
    REQUIRED_ATTRIBUTES = {
        'DF_COLS_ATTR': dict
    }

    @abstractmethod
    def __init__(self):
        self.db_mngr = DatabaseManager()
        self._data = pd.DataFrame()

    @property
    def data(self):
        """pd.DataFrame: Data."""
        return self._data

    @property
    def columns(self):
        """pd.Index: DataFrame columns name"""
        return self.data.columns

    def __getattr__(self, item):
        try:
            if item in self.DF_COLS_ATTR.keys():
                return self.data[item]

            else:
                return super().__getattribute__(item)

        except KeyError:
            raise dvasError(f"Valid keys are: {self.columns}")

    def __setattr__(self, item, val):
        try:
            if item == 'data':

                # Check that I have all the columns I need in the input, with the proper format.
                val = self._test_cols(val)

                # Set index
                val.set_index(
                    [key for key, val in self.DF_COLS_ATTR.items() if val['idx'] is True],
                    drop=False, inplace=True
                )
                val.sort_index(inplace=True)

                # If so, set things up
                self._data = val[list(self.DF_COLS_ATTR.keys())]

            elif item in self.DF_COLS_ATTR.keys():
                # Test input value
                assert isinstance(val, pd.Series)
                value = self._test_cols(pd.DataFrame(val, columns=[item,]), cols_key=[item])

                # Update value
                self._data[item].update(value[item])

            else:
                super().__setattr__(item, val)

        except KeyError:
            raise dvasError(f"Valid keys are: {self.columns}")
        except AssertionError:
            raise dvasError(f"Value must be a pd.Series")

    def __delattr__(self, item):
        raise dvasError(f"Can't delete attribute '{item}'")

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)

    @abstractmethod
    def copy(self):
        """Copy method"""

    @classmethod
    def _test_cols(cls, val, cols_key=None):
        """ Test that all the required columns are present, with the correct type.

        Args:
           val (pandas.DataFrame): The data to fill the Profile.
           cols_key (list of str, `optional`): Column key to test

        Raises:
            dvasError: Missing data column.
            dvasError: Wrong data type.
        """

        # Init
        if cols_key is None:
            cols_key = cls.DF_COLS_ATTR.keys()

        # Test val columns
        for key in filter(lambda x: x in cols_key, cls.DF_COLS_ATTR.keys()):

            # Test column name
            if key not in val.columns:
                raise dvasError('Required column not found: %s' % key)

            # Test column type
            if ~val[key].apply(type).apply(
                    issubclass, args=(cls.DF_COLS_ATTR[key]['test']+(type(None),),)
            ).all():
                raise dvasError(
                    "Wrong data type for '%s': I need %s but you gave me %s" %
                    (key, cls.DF_COLS_ATTR[key]['test'], val[key].dtype)
                )

            # Convert
            if cls.DF_COLS_ATTR[key]['type']:
                val[key] = val[key].astype(cls.DF_COLS_ATTR[key]['type'])

        return val


class Profile(ProfileAbstract):
    """Base Profile class for atmospheric measurements. Requires only some measured values,
    together with their corresponding altitudes and flags.

    The data is stored in a pandas DataFrame with column labels:
      - 'alt' (float)
      - 'val' (float)
      - 'flg' (Int64)

    The same format is expected as input.

    """

    FLAG_BIT_NM = Flag.bit_number.name
    FLAG_ABBR_NM = Flag.flag_abbr.name
    FLAG_DESC_NM = Flag.flag_desc.name

    # The column names for the pandas DataFrame
    DF_COLS_ATTR = {
        'alt': {'test': FLOAT_TEST, 'type': np.float, 'idx': True},
        'val': {'test': FLOAT_TEST, 'type': np.float, 'idx': False},
        'flg': {'test': INT_TEST, 'type': 'Int64', 'idx': False}
    }

    def __init__(self, event, data=None):
        """ Profile Constructor.

        Args:
            event (int): Event id
            data (pd.DataFrame, optional): The profile values in a pandas DataFrame.
               Default to None.

        """
        super(Profile, self).__init__()

        # Set attributes
        if data is not None:
            self.data = data
        else:
            self.data = pd.DataFrame(
                {key: np.array([], dtype=val['type']) for key, val in self.DF_COLS_ATTR.items()}
            )
        self._event = event
        self._flags_abbr = {arg[self.FLAG_ABBR_NM]: arg for arg in self.db_mngr.get_flags()}

    @property
    def event(self):
        """Event: Corresponding data event metadata"""
        return self._event

    @property
    def flags_abbr(self):
        """dict: Flag abbr, description and bit position."""
        return self._flags_abbr

    @property
    def alt(self):
        """pd.Series: Corresponding data altitude"""
        return super().__getattr__('alt')

    @property
    def val(self):
        """pd.Series: Corresponding data 'val'"""
        return super().__getattr__('val')

    @property
    def flg(self):
        """pd.Series: Corresponding data 'flag'"""
        return super().__getattr__('flg')

    def __str__(self):
        return f"event: {self.event}\n{super().__str__()}"

    def copy(self):
        return self.__class__(self.event, self.data.copy(deep=True))

    def _get_flg_bit_nbr(self, abbr):
        """Get bit number corresponding to given flag abbr"""
        return self.flags_abbr[abbr][self.FLAG_BIT_NM]

    def set_flg(self, abbr, set_val, index=None):
        """Set flag values to True/False.

        Args:
            abbr (str): flag name
            set_val (bool): Turn on/off the flag. Defaults to True.
            index (pd.Index, optional): Specific Profile elements to set. Default to None (=all).

        """

        # Define
        def set_to_true(x):
            """Set bit to True"""
            if np.isnan(x):
                out = (1 << self._get_flg_bit_nbr(abbr))
            else:
                out = int(x) | (1 << self._get_flg_bit_nbr(abbr))

            return out

        def set_to_false(x):
            """Set bit to False"""
            if np.isnan(x):
                out = 0
            else:
                out = int(x) & ~(1 << self._get_flg_bit_nbr(abbr))

            return out

        # Init
        if index is None:
            index = self.data.index

        # Set bit
        if set_val is True:
            self.flg = self.flg.loc[index].apply(set_to_true)
        else:
            self.flg = self.flg.loc[index].apply(set_to_false)

    def is_flagged(self, abbr):
        """Check if a specific flag tag is set.

        Args:
            abbr (str): flag name

        Returns:
            pd.Series: Series of int, with 1's where the requested flag name is True.

        """
        bit_nbr = self._get_flg_bit_nbr(abbr)
        return self.flg.apply(lambda x: (x >> bit_nbr) & 1)


class RSProfile(Profile):
    """ Child Profile class for *basic radiosonde* atmospheric measurements.
    Requires some measured values, together with their corresponding measurement times since launch,
    altitudes, and flags.

    The data is stored in a pandas DataFrame with column labels:
    - 'alt' (float)
    - 'tdt' (timedelta64[ns])
    - 'val' (float)
    - 'flag' (Int64)

    The same format is expected as input.

    """

    # The column names for the pandas DataFrame
    DF_COLS_ATTR = {
        'alt': {'test': FLOAT_TEST, 'type': np.float, 'idx': False},
        'tdt': {'test': TIME_TEST, 'type': 'timedelta64[ns]', 'idx': True},
        'val': {'test': FLOAT_TEST, 'type': np.float, 'idx': False},
        'flg': {'test': INT_TEST, 'type': 'Int64', 'idx': False},
    }

    @property
    def tdt(self):
        """pd.Series: Corresponding data time delta since launch"""
        return super().__getattr__('tdt')


class GDPProfile(RSProfile):
    """ Child RSProfile class for *GDP-like* radiosonde atmospheric measurements.
    Requires some measured values, together with their corresponding measurement times since launch,
    altitudes, flags, as well as 4 distinct types uncertainties:

      - 'uc_n' : True uncorrelated uncertainties.
      - 'uc_r' : Rig "uncorrelated" uncertainties.
      - 'uc_s' : Spatial-correlated uncertainties.
      - 'uc_t' : Temporal correlated uncertainties.

    The property "uc_tot" returns the total uncertainty, and is prodvided for convenience.

    The data is stored in a pandas DataFrame with column labels:
    - 'alt' (float)
    - 'tdt' (timedelta64[ns])
    - 'val' (float)
    - 'ucn' (float)
    - 'ucr' (float)
    - 'ucs' (float)
    - 'uct' (float)
    - 'flg' (Int64)

    The same format is expected as input.

    """

    # The column names for the pandas DataFrame
    DF_COLS_ATTR = dict(
        **RSProfile.DF_COLS_ATTR,
        **{
            'ucn': {'test': FLOAT_TEST, 'type': np.float, 'idx': False},
            'ucr': {'test': FLOAT_TEST, 'type': np.float, 'idx': False},
            'ucs': {'test': FLOAT_TEST, 'type': np.float, 'idx': False},
            'uct': {'test': FLOAT_TEST, 'type': np.float, 'idx': False},
          }
    )

    @property
    def uc_tot(self):
        """ Computes the total uncertainty from the individual components.

        Returns:
            pd.Series: uc_tot = np.sqrt(uc_n**2 + uc_r**2 + uc_s**2 + uc_t**2)
        """

        return np.sqrt(self.ucn.fillna(0)**2 +
                       self.ucr.fillna(0)**2 +
                       self.ucs.fillna(0)**2 +
                       self.uct.fillna(0)**2)
