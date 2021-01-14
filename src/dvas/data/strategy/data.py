"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data manager classes used in dvas.data.data.ProfileManager

"""

# Import from external packages
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np
import pandas as pd

# Import from current package
from ...database.model import Flag as TableFlag
from ...database.database import DatabaseManager, InfoManager
from ...errors import ProfileError
from ...helper import RequiredAttrMetaClass
from ..hardcoded import PRF_REF_INDEX_NAME

# Define
INT_TEST = (np.int64, np.int, int, type(pd.NA))
FLOAT_TEST = (np.float, float) + INT_TEST
TIME_TEST = FLOAT_TEST + (pd.Timedelta,)


class MPStrategyAC(metaclass=ABCMeta):
    """Abstract class (AC) for a multiprofile (MP) strategy"""

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute strategy method"""


class ProfileAC(metaclass=RequiredAttrMetaClass):
    """Abstract Profile class"""

    # Specify required attributes
    # DF_COLS_ATTR dictionary items:
    # - 'test': types to test (tuple of type)
    # - 'type': type conversion (type|None)
    # - 'index': use as index (bool)
    REQUIRED_ATTRIBUTES = {
        'DF_COLS_ATTR': dict,
    }

    DF_COLS_ATTR = None

    @abstractmethod
    def __init__(self):
        self._data = pd.DataFrame()

    @property
    def data(self):
        """pd.DataFrame: Data."""
        return self._data

    @data.setter
    def data(self, value):
        setattr(self, 'data', value)

    @property
    def columns(self):
        """pd.Index: DataFrame columns name"""
        return self.data.columns

    @classmethod
    def reset_data_index(cls, val):
        """Return the data with reset index

        Args:
            val (pandas.DataFrame): DataFrame with index to be reset

        Returns:
            pandas.DataFrame

        """
        val = val.reset_index(inplace=False)
        val.index.name = PRF_REF_INDEX_NAME
        return val[sorted(cls.DF_COLS_ATTR.keys())]

    @classmethod
    def set_data_index(cls, val):
        """Return the data with reset index

        Args:
            val (pandas.DataFrame): DataFrame with index to be reset

        Returns:
            pandas.DataFrame

        """
        val = cls.reset_data_index(val)
        val = val.set_index(
            cls.get_index_attr(), drop=True, append=True, inplace=False
        )

        return val

    @classmethod
    def get_index_attr(cls):
        """Get index attributes

        Returns:
            list

        """
        return sorted([
            key for key, val in
            cls.DF_COLS_ATTR.items() if val['index'] is True
        ])

    @classmethod
    def get_col_attr(cls):
        """Get columns attributes

                Returns:
                    list

                """
        return sorted([
            key for key, val in
            cls.DF_COLS_ATTR.items() if val['index'] is False
        ])

    def __getattr__(self, item):
        try:
            if item in self.get_col_attr():
                return self.data[item]

            if item in self.get_index_attr():
                # fpavogt, 2020-12-18: what follows returns an Index. We may need to revise this
                # at some point. Or not. But as it stands, it's not designed to be concatenated
                # with anythinbg else.
                return self.data.index.get_level_values(item)

            return super().__getattribute__(item)

        except KeyError:
            raise ProfileError(f"Valid keys are: {self.columns}")

    def __setattr__(self, item, val):
        try:
            if item == 'data':

                # Check that I have all the columns I need in the input, with the proper format.
                val = self._prepare_df(val)

                # Set index
                val = self.set_data_index(val)

                # If so, set things up
                self._data = val[self.get_col_attr()]

            elif item in self.get_col_attr():

                # Prepare value
                assert isinstance(val, pd.Series)
                value = self._prepare_df(pd.DataFrame(val, columns=[item,]), cols_key=[item])

                # Update value
                self._data[item].update(value[item])

            elif item in self.get_index_attr():
                raise AttributeError(f"{item} is an index and can not be set")

            else:
                super().__setattr__(item, val)

        except AttributeError as exc:
            raise AttributeError(exc)

        except (KeyError, ProfileError):
            raise ProfileError(
                f"Valid keys are: {list(self.DF_COLS_ATTR.keys())}. " +\
                f"You gave {val.name if isinstance(val, pd.Series) else val.columns}"
            )
        except AssertionError:
            raise ProfileError(f"Value must be a pd.Series")

    def __delattr__(self, item):
        raise ProfileError(f"Can't delete attribute '{item}'")

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)

    @abstractmethod
    def copy(self):
        """Copy method"""

    @classmethod
    def _prepare_df(cls, val, cols_key=None):
        """ Prepare the DataFrame. Test that all the required columns are present,
        with the correct type and convert the column type.

        Args:
           val (pandas.DataFrame): The data to fill the Profile.
           cols_key (list of str, `optional`): Column key to test

        Returns:
            pandas.DataFrame

        Raises:
            ProfileError: Missing data column.
            ProfileError: Wrong data type.
        """

        # Init
        if cols_key is None:
            cols_key = cls.DF_COLS_ATTR.keys()

        # Test val columns
        for key in filter(lambda x: x in cols_key, cls.DF_COLS_ATTR.keys()):

            # If the key is an index, then get it back out as a normal column.
            # This is a lot easier than having to handle the distinct cases of Index vs Column
            # in what follows.
            if key in val.index.names:
                # Note: do not even think about using inplace=True in the next code line.
                # Dark things will happen if you do. fpvogt - 2020-12-16
                val = val.reset_index(key)

            # Test column name. If it is missing, raise an error
            if key not in val.columns:
                raise ProfileError('Required column not found: %s' % key)

            # ... to test if they all have the proper type.
            if ~val[key].apply(type).apply(
                    issubclass, args=(cls.DF_COLS_ATTR[key]['test']+(type(None),),)).all():
                raise ProfileError("Wrong data type for '%s': I need %s but you gave me %s" %
                                   (key, cls.DF_COLS_ATTR[key]['test'], val[key].dtype))

            # Convert
            if key in val.columns and cls.DF_COLS_ATTR[key]['type']:
                val[key] = val[key].astype(cls.DF_COLS_ATTR[key]['type'])

        return val


class Profile(ProfileAC):
    """Base Profile class for atmospheric measurements. Requires only some measured values,
    together with their corresponding altitudes and flags.

    The data is stored in a pandas DataFrame with column labels:
      - 'alt' (float)
      - 'val' (float)
      - 'flg' (Int64)

    The same format is expected as input.

    """

    FLAG_BIT_POS_NM = TableFlag.bit_pos.name
    FLAG_NAME_NM = TableFlag.flag_name.name
    FLAG_DESC_NM = TableFlag.flag_desc.name

    # The column names for the pandas DataFrame
    DF_COLS_ATTR = {
        'val': {'test': FLOAT_TEST, 'type': np.float, 'index': False},
        'alt': {'test': FLOAT_TEST, 'type': np.float, 'index': True},
        'flg': {'test': FLOAT_TEST, 'type': 'Int64', 'index': False}
    }

    def __init__(self, info, data=None):
        """ Profile Constructor.

        Args:
            info (InfoManager): Data information
            data (pd.DataFrame, optional): The profile values in a pandas DataFrame.
               Default to None.

        """
        super(Profile, self).__init__()

        # Init
        db_mngr = DatabaseManager()

        # Test info
        try:
            assert isinstance(info, InfoManager)
        except AssertionError:
            raise ProfileError('Bad argument type. Must be an InfoManager.')

        # Set attributes
        if data is not None:
            self.data = data
        else:
            self.data = pd.DataFrame(
                {key: np.array([], dtype=val['type']) for key, val in self.DF_COLS_ATTR.items()}
            )

        self._info = info
        self._flags_abbr = {arg[self.FLAG_NAME_NM]: arg for arg in db_mngr.get_flags()}

    @property
    def info(self):
        """InfoManager: Corresponding data info"""
        return self._info

    @property
    def flags_abbr(self):
        """dict: Flag name, description and bit position."""
        return self._flags_abbr

    @property
    def alt(self):
        """pd.Series: Corresponding data altitude"""
        return super().__getattr__('alt')

    @property
    def val(self):
        """pd.Series: Corresponding data 'val'"""
        return super().__getattr__('val')

    @val.setter
    def val(self, value):
        setattr(self, 'val', value)

    @property
    def flg(self):
        """pd.Series: Corresponding data 'flag'"""
        return super().__getattr__('flg')

    @flg.setter
    def flg(self, value):
        setattr(self, 'flg', value)

    def __str__(self):
        return f"info: {self.info}\n{super().__str__()}"

    def copy(self):
        return self.__class__(
            deepcopy(self.info), self.reset_data_index(self.data.copy(deep=True))
        )

    def _get_flg_bit_nbr(self, abbr):
        """Get bit number corresponding to given flag name"""
        return self.flags_abbr[abbr][self.FLAG_BIT_POS_NM]

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
        """Check if a specific flag name is set.

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
        'alt': {'test': FLOAT_TEST, 'type': np.float, 'index': True},
        'tdt': {'test': TIME_TEST, 'type': 'timedelta64[ns]', 'index': True},
        'val': {'test': FLOAT_TEST, 'type': np.float, 'index': False},
        'flg': {'test': FLOAT_TEST, 'type': 'Int64', 'index': False},
    }

    @property
    def tdt(self):
        """pd.Series: Corresponding data time delta since launch"""
        return super().__getattr__('tdt')


class GDPProfile(RSProfile):
    """ Child RSProfile class for *GDP-like* radiosonde atmospheric measurements.
    Requires some measured values, together with their corresponding measurement times since launch,
    altitudes, flags, as well as 4 distinct types uncertainties:

      - 'uc_r' : Rig "uncorrelated" uncertainties.
      - 'uc_s' : Spatial-correlated uncertainties.
      - 'uc_t' : Temporal correlated uncertainties.
      - 'uc_u' : True uncorrelated uncertainties.

    The property "uc_tot" returns the total uncertainty, and is prodvided for convenience.

    The data is stored in a pandas DataFrame with column labels:
    - 'alt' (float)
    - 'tdt' (timedelta64[ns])
    - 'val' (float)
    - 'ucr' (float)
    - 'ucs' (float)
    - 'uct' (float)
    - 'ucu' (float)
    - 'flg' (Int64)

    The same format is expected as input.

    """

    # The column names for the pandas DataFrame
    DF_COLS_ATTR = dict(
        **RSProfile.DF_COLS_ATTR,
        **{
            'ucr': {'test': FLOAT_TEST, 'type': np.float, 'index': False},
            'ucs': {'test': FLOAT_TEST, 'type': np.float, 'index': False},
            'uct': {'test': FLOAT_TEST, 'type': np.float, 'index': False},
            'ucu': {'test': FLOAT_TEST, 'type': np.float, 'index': False},
          }
    )

    @property
    def ucr(self):
        """pd.Series: Corresponding data time delta since launch"""
        return super().__getattr__('ucr')

    @ucr.setter
    def ucr(self, value):
        setattr(self, 'ucr', value)

    @property
    def ucs(self):
        """pd.Series: Corresponding data time delta since launch"""
        return super().__getattr__('ucs')

    @ucs.setter
    def ucs(self, value):
        setattr(self, 'ucs', value)

    @property
    def uct(self):
        """pd.Series: Corresponding data time delta since launch"""
        return super().__getattr__('uct')

    @uct.setter
    def uct(self, value):
        setattr(self, 'uct', value)

    @property
    def ucu(self):
        """pd.Series: Corresponding data time delta since launch"""
        return super().__getattr__('ucu')

    @ucu.setter
    def ucu(self, value):
        setattr(self, 'ucu', value)

    @property
    def uc_tot(self):
        """ Computes the total uncertainty from the individual components.

        Returns:
            pd.Series: uc_tot = np.sqrt(uc_r**2 + uc_s**2 + uc_t**2 + uc_u**2)
        """

        out = np.sqrt(self.ucr.fillna(0)**2 + self.ucs.fillna(0)**2 +
                      self.uct.fillna(0)**2 + self.ucu.fillna(0)**2)
        # For those cases where all I have is NaN, then return a NaN (and not a 0.)
        out[self.data[['ucr', 'ucs', 'uct', 'ucu']].isna().all(axis=1)] = np.nan
        # Make sure to give a proper name to the Series
        return out.rename('uc_tot')
