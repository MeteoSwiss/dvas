"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data manager classes used in dvas.data.data.ProfileManager

"""

# Import from external packages
from copy import deepcopy
from abc import ABCMeta
import numpy as np
import pandas as pd

# Import from current package
from ...database.model import Flag
from ...database.database import DatabaseManager
from ...dvas_logger import dvasError

class Profile(metaclass=ABCMeta):
    """Base Profile class for atmospheric measurements. Requires only some measured values,
    together with their corresponding altitudes and flags.

    The data is stored in a pandas DataFrame with column labels:
      - 'alt' (float)
      - 'val' (float)
      - 'flg' (int64)

    The same format is expected as input.

    """

    FLAG_BIT_NM = Flag.bit_number.name
    FLAG_ABBR_NM = Flag.flag_abbr.name
    FLAG_DESC_NM = Flag.flag_desc.name

    # The column names for the pandas DataFrame
    PD_COLS = {'alt': np.float, 'val': np.float, 'flg': np.int64}

    db_mngr = DatabaseManager()

    def __init__(self, event_mngr, data=None):
        """ Profile Constructor.

        Args:
            event_mngr (int): Event id
            data (pd.DataFrame, optional): The profile values in a pandas DataFrame.
               Default to None.

        """

        # Set attributes
        self._data = pd.concat([pd.Series(dtype=self.PD_COLS[item]) for item in self.PD_COLS],
                               axis=1)
        self._data.columns = self.PD_COLS.keys()
        self._event_mngr = event_mngr
        self._flags_abbr = {arg[self.FLAG_ABBR_NM]: arg for arg in self.db_mngr.get_flags()}

        # Set the data if applicable
        if data is not None:
            self.data = data

    @property
    def data(self):
        """pd.DataFrame: Data."""
        return self._data

    @data.setter
    def data(self, val):
        """ Setter for the data"""

        # Check that I have all the columns I need in the input, with the proper format.
        self._test_cols(val)

        # If so, set things up.
        self._data = val[list(self.PD_COLS.keys())]

    @property
    def event_mngr(self):
        """EventManager: Corresponding data event manager"""
        return self._event_mngr

    @property
    def flags_abbr(self):
        """dict: Flag abbr, description and bit position."""
        return self._flags_abbr

    @property
    def alt(self):
        """pd.Series: Corresponding data altitude"""
        return self._data['alt']

    @property
    def val(self):
        """pd.Series: Corresponding data 'val'"""
        return self._data['val']

    # fpavogt, 05.11.2020: Disabling this for now. The new spirit has all the data in a DataFrame.
    # We should avoid encouraging the modifcation of single columns directly to remind the user
    # that alt, val and flags should always go together ... at least for now ?
    #@value.setter
    #def value(self, val):
    #
    #    # Test arg
    #    self._test_index(val.index)
    #
    #    # Set series name/dtype
    #    val.name = 'value'
    #    val = val.astype('float')
    #
    #    if len(self) == 0:
    #        self.data['value'] = val
    #    else:
    #        self.data.update(val)

    @property
    def flg(self):
        """pd.Series: Corresponding data 'flag'"""
        return self._data['flg']

    @flg.setter
    def flg(self, val):
        """ Flag setter function.

        Args:
           val: pd.DataFrame with a column 'flag' (Int64)
        """

        if val.name != 'flg':
            raise dvasError('Flag column not found.')

        # Force the dtype (Int64 uses special NaN for integers)
        val = val.astype('Int64')

        # Set the data
        if len(self) == 0:
            self._data['flg'] = val
        else:
            self._data.update(val)

    def copy(self):
        """Copy method"""
        return deepcopy(self)

    def __len__(self):
        return len(self.data)

    def _get_flg_bit_nbr(self, abbr):
        """Get bit number corresponding to given flag abbr"""
        return self.flags_abbr[abbr][self.FLAG_BIT_NM]

    def toogle_flg(self, abbr, set_val=True, index=None):
        """Toogle flag values on/off.

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

    def _test_cols(self, val):
        """ Test that all the required columns are present, with the correct type.

        Args:
           val (pandas.DataFrame): The data to fill the Profile.

        Raises:
            dvasError: Missing data column.
            dvasError: Wrong data type.
        """

        for item in self.PD_COLS:
            if item not in val.columns:
                raise dvasError('Required column not found: %s' % (item))

            if pd.Series(dtype=self.PD_COLS[item]).dtype != val[item].dtype:
                raise dvasError('Wrong data type for %s: I need %s but you gave me %s' %
                                (item, self.PD_COLS[item], val[item].dtype))


class RSProfile(Profile):
    """ Child Profile class for *basic radiosonde* atmospheric measurements.
    Requires some measured values, together with their corresponding measurement times since launch,
    altitudes, and flags.

    The data is stored in a pandas DataFrame with column labels:
    - 'alt' (float)
    - 'tdt' (timedelta64[ns])
    - 'val' (float)
    - 'flag' (int64)

    The same format is expected as input.

    """

    # The column names for the pandas DataFrame
    PD_COLS = {'alt': np.float, 'tdt': 'timedelta64[ns]', 'val': np.float, 'flg': np.int64}

    @property
    def tdt(self):
        """pd.Series: Corresponding data time delta since launch"""
        return self._data['tdt']


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
    - 'tdt' (datetime.datetime)
    - 'val' (float)
    - 'ucn' (float)
    - 'ucr' (float)
    - 'ucs' (float)
    - 'uct' (float)
    - 'flg' (int64)

    The same format is expected as input.

    """

    # The column names for the pandas DataFrame
    PD_COLS = {'alt': np.float, 'tdt': 'timedelta64[ns]', 'val': np.float, 'ucn': np.float,
               'ucr': np.float, 'ucs': np.float, 'uct': np.float, 'flg': np.int64}

    @property
    def uc_tot(self):
        """ Computes the total uncertainty from the individual components.

        Returns:
            pd.Series: uc_tot = np.sqrt(uc_n**2 + uc_r**2 + uc_s**2 + uc_t**2)
        """

        return np.sqrt(self.data.ucn.fillna(0)**2 +
                       self.data.ucr.fillna(0)**2 +
                       self.data.ucs.fillna(0)**2 +
                       self.data.uct.fillna(0)**2)
