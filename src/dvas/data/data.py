"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data management

"""

# Import from external packages
from abc import abstractmethod
import pandas as pd

# Import from current package
from .strategy.data import Profile, RSProfile, GDPProfile

from .strategy.load import LoadProfileStrategy, LoadRSProfileStrategy, LoadGDPProfileStrategy

from .strategy.sort import SortProfileStrategy

from .strategy.plot import PlotStrategy, RSPlotStrategy, GDPPlotStrategy

from .strategy.rebase import RebaseStrategy
from .strategy.resample import ResampleRSStrategy, ResampleGDPStrategy
from .strategy.save import SaveDataStrategy

from ..database.database import DatabaseManager
from ..database.model import Parameter as TableParameter
from ..helper import RequiredAttrMetaClass
from ..helper import deepcopy
from ..helper import get_class_public_attr

from ..errors import DvasError, DBIOError

from ..hardcoded import TAG_RAW_NAME


# Loading strategies
load_prf_stgy = LoadProfileStrategy()
load_rsprf_stgy = LoadRSProfileStrategy()
load_gdpprf_stgy = LoadGDPProfileStrategy()

# Plotting strategies
plt_prf_stgy = PlotStrategy()
plt_rsprf_stgy = RSPlotStrategy()
plt_gdpprf_stgy = GDPPlotStrategy()

# Rebasing strategies
rebase_prf_stgy = RebaseStrategy()

# Resampling strategies
resample_rsprf_stgy = ResampleRSStrategy()
resample_gdpprf_stgy = ResampleRSStrategy() # TODO

# Other strategies
sort_prf_stgy = SortProfileStrategy()
save_prf_stgy = SaveDataStrategy()


# TODO
#  Create a factory (in terms of design patterns) to easily build MultiProfiles
class MutliProfileAC(metaclass=RequiredAttrMetaClass):
    """Abstract MultiProfile class"""

    # Specify required attributes
    # _DATA_TYPES:
    # - type (Profile|RSProfile|GDPProfile|...)
    REQUIRED_ATTRIBUTES = {
        '_DATA_TYPES': type
    }

    #: type: Data type
    _DATA_TYPES = None

    _DATA_EMPTY = []
    _DB_VAR_EMPTY = {}

    @abstractmethod
    def __init__(self, load_stgy=None, sort_stgy=None, save_stgy=None, plot_stgy=None,
                 rebase_stgy=None):

        # Init attributes
        self._load_stgy = load_stgy
        self._sort_stgy = sort_stgy
        self._save_stgy = save_stgy
        self._plot_stgy = plot_stgy
        self._rebase_stgy = rebase_stgy

        self._profiles = self._DATA_EMPTY
        self._db_variables = self._DB_VAR_EMPTY

    @property
    def profiles(self):
        """list of Profile"""
        return self._profiles

    @property
    def db_variables(self):
        """dict: Correspondence between DataFrame and DB parameter"""
        return self._db_variables

    @property
    def var_info(self):
        """dict: Variable informations"""

        # Define
        db_mngr = DatabaseManager()

        # Query parameter info
        qry_res = db_mngr.get_table(
            TableParameter,
            search={
                'where': TableParameter.prm_name.in_(
                    [val for val in self.db_variables.values() if val]
                )
            }
        )

        # Swap db variables dict
        var_db = {val: key for key, val in self.db_variables.items() if val}

        # Set output
        out = {
            var_db[res[TableParameter.prm_name.name]]: {
                TableParameter.prm_name.name: res[TableParameter.prm_name.name],
                TableParameter.prm_desc.name: res[TableParameter.prm_desc.name],
                TableParameter.prm_unit.name: res[TableParameter.prm_unit.name],
            } for res in qry_res
        }

        return out

    @property
    def info(self):
        """list of ProfileManger info: Data info"""
        return [arg.info for arg in self.profiles]

    @deepcopy
    def rm_info_tags(self, val):
        """Remove some tags from all info tag lists.

        Args:
            val (str|list of str): Tag value(s) to remove

        """
        for i in range(len(self)):
            self.profiles[i].info.rm_tags(val)

    @deepcopy
    def add_info_tags(self, val):
        """Add tag from all info tags

        Args:
            val (str|list of str): Tag values to add.

        """
        for i in range(len(self)):
            self.profiles[i].info.add_tags(val)

    def __len__(self):
        return len(self.profiles)

    def copy(self):
        """Return a deep copy of the object"""
        obj = self.__class__()
        obj._db_variables = self.db_variables.copy()
        obj._profiles = [arg.copy() for arg in self.profiles]
        return obj

    @deepcopy
    def load_from_db(self, *args, **kwargs):
        """Load data from the database.

        Args:
            *args: positional arguments
            **kwargs: key word arguments

        """

        # Call the appropriate Data strategy
        data, db_df_keys = self._load_stgy.execute(*args, **kwargs)

        # Test data len
        if not data:
            raise DBIOError('Load empty data')

        # Update
        self.update(db_df_keys, data)

    @deepcopy
    def sort(self):
        """Sort method

        """

        # Sort
        data = self._sort_stgy.execute(self.profiles)

        # Load
        self.update(self.db_variables, data)

    def save_to_db(self, add_tags=None, rm_tags=None, prms=None):
        """Save method to store the *entire* content of the Multiprofile
        instance back into the database with an updated set of tags.

        Args:
            add_tags (list of str, optional): list of tags to add to the entity when inserting it
                into the database. Defaults to None.
            rm_tags (list of str, optional): list of *existing* tags to remove
                from the entity before inserting ot into the database. Defaults to None.
            prms (list of str, optional): list of column names to save to the
                database. Defaults to None (= save all possible parameters).

        Notes:
            The 'raw' tag will always be removed and the 'derived' tag will
            always be added by default when saving anything into the database.

        """

        # Init
        obj = self.copy()

        # Add tags
        if add_tags is not None:
            obj.add_info_tags(add_tags)

        # Remove tag RAW
        rm_tags = [TAG_RAW_NAME] if rm_tags is None else rm_tags + [TAG_RAW_NAME]

        # Remove tags
        obj.rm_info_tags(rm_tags)

        # Restructure the parameters into a dict, to be consistent with the rest of the class.
        if prms is None:
            prms = list(self.db_variables.keys())

        # Call save strategy
        self._save_stgy.execute(obj, prms)

    # TODO: implement an "export" function that can export specific DataFrame columns back into
    #  the database under new variable names ?
    # THis may be confusing. WOuldn't it be sufficient to change the keys in db_variables and then
    # use the existing "save_to_db" method ?

    def update(self, db_df_keys, data):
        """Update the whole Multiprofile list with new Profiles.

        Args:
            db_df_keys (dict): Relationship between database parameters and
                Profile.data columns.
            data (list of Profile): Data

        """

        # Check input type
        assert isinstance(data, list), "Was expecting a list, not: %s" % (type(data))

        # Test data if not empty
        if data:

            # Check input value type
            assert all(
                [isinstance(arg, self._DATA_TYPES) for arg in data]
            ), f"Wrong data type: I need {self._DATA_TYPES} but you gave me {[type(arg) for arg in data]}"

            # Check db keys
            assert (
                set(db_df_keys.keys()) == set(data[0].get_col_attr() + data[0].get_index_attr())
            ), f"Key {db_df_keys} does not match data columns"

        else:
            data = self._DATA_EMPTY
            db_df_keys = self._DB_VAR_EMPTY

        # Update
        self._db_variables = db_df_keys
        self._profiles = data

    def append(self, db_df_keys, val):
        """Append method

        Args:
            db_df_keys (dict): Relationship between database parameters and
                Profile.data columns.
            val (Profile): Data

        """

        self.update(db_df_keys, self.profiles + [val])

    def get_prms(self, prm_list=None):
        """ Convenience getter to extract specific columns from the DataFrames and/or class
        properties of all the Profile instances.

        Only column/property names are allowed. Specifying only index names will raise a DvasError.

        Args:
            prm_list (str|list of str, optional): names of the columns(s) to extract from all the
                Profile DataFrames. Defaults to None (=return all the columns from the DataFrame).

        Returns:
            dict of list of DataFrame: idem to self.profiles, but with only the requested data.

        Raises:
            DvasError: if prm_list only contains the names of Indices.

        """

        if prm_list is None:
            prm_list = list(self.db_variables.keys())

        if isinstance(prm_list, str):
            # Assume the user forgot to put the key into a list.
            prm_list = [prm_list]

        # Remove any prm that is an index name
        prm_list = [prm for prm in prm_list
                    if not any([prm in arg.get_index_attr() for arg in self.profiles])]

        # Check that I still have something valid to extract !
        if len(prm_list) == 0:
            raise DvasError("Ouch ! Invalid column name(s). Did you only specify index name(s) ?")

        # Select data
        try:
            out = [
                pd.concat(
                    [getattr(arg, prm) for prm in prm_list],
                    axis=1, ignore_index=False
                )
                for arg in self.profiles
            ]
        except AttributeError:
            raise DvasError(f"Unknown parameter/attribute name in {prm_list}")

        return out

    def get_info(self, prm=None):
        """ Convenience function to extract Info from all the Profile instances.

        Args:
            prm (str, `optional`): Info attribute to extract. Default to None.

        Returns:
            dict of list: idem to self.profiles, but with only the requested metadata.

        """

        if prm:
            out = [getattr(info, prm) for info in self.info]
        else:
            out = [get_class_public_attr(info) for info in self.info]

        return out

    def plot(self, **kwargs):
        """ Plot method

        Args:
            **kwargs: Keyword arguments to be passed down to the plotting function.

        Returns:
            None

        """

        self._plot_stgy.execute(self, **kwargs)

    @deepcopy
    def rebase(self, new_lengths, shifts=None):
        """ Rebase method, which allows to map Profiles on new set of integer indices.

        This will move the values around, including the non-integer indices (i.e. anything
        other than '_idx') if applicable.

        Args:
            new_lengths (int|list of int): The length of the DataFrame to rebase upon.
                If specifiying an int, the same length will be applied to all Profiles. Else, the
                list should specify a length for each Profile.
            shifts (int|list of int, optional): row n of the existing data will become row n+shift.
                If specifiying an int, the same shift will be applied to all Profiles. Else, the
                list should specify a shift for each Profile. Defaults to None (=no shift).

        """

        data = self._rebase_stgy.execute(self.profiles, new_lengths, shifts=shifts)

        self.update(self.db_variables, data)


class MultiProfile(MutliProfileAC):
    """Multi profile base class, designed to handle multiple Profile."""

    #: type: supported Profile Types
    _DATA_TYPES = Profile

    def __init__(self):
        super().__init__(
            load_stgy=load_prf_stgy, sort_stgy=sort_prf_stgy,
            save_stgy=save_prf_stgy, plot_stgy=plt_prf_stgy, rebase_stgy=rebase_prf_stgy,
        )


class MultiRSProfileAC(MutliProfileAC):
    """Abstract MultiRSProfile class"""

    @abstractmethod
    def __init__(self, load_stgy=None, sort_stgy=None,
                 save_stgy=None, plot_stgy=None, rebase_stgy=None, resample_stgy=None,
                 ):
        super().__init__(load_stgy=load_stgy, sort_stgy=sort_stgy, save_stgy=save_stgy,
                         plot_stgy=plt_prf_stgy, rebase_stgy=rebase_prf_stgy)

        self._resample_stgy = resample_rsprf_stgy

    @deepcopy
    def resample(self, freq='1s'):
        """Resample the profiles (one-by-one) onto regular timesteps using linear interpolation.

        Args:
            freq (str): see pandas.timedelta_range(). Defaults to '1s'.

        """

        data = self._resample_stgy.execute(self.profiles, freq=freq)
        self.update(self.db_variables, data)

class MultiRSProfile(MultiRSProfileAC):
    """Multi RS profile manager, designed to handle multiple RSProfile instances."""

    _DATA_TYPES = RSProfile

    def __init__(self):
        super().__init__(
            load_stgy=load_rsprf_stgy, sort_stgy=sort_prf_stgy,
            save_stgy=save_prf_stgy, plot_stgy=plt_prf_stgy, rebase_stgy=rebase_prf_stgy,
        )

        self._resample_stgy = resample_rsprf_stgy


class MultiGDPProfile(MultiRSProfileAC):
    """Multi GDP profile manager, designed to handle multiple GDPProfile instances."""

    _DATA_TYPES = GDPProfile

    def __init__(self):
        super().__init__(
            load_stgy=load_gdpprf_stgy, sort_stgy=sort_prf_stgy,
            save_stgy=save_prf_stgy, plot_stgy=plt_prf_stgy, rebase_stgy=rebase_prf_stgy,
        )

        self._resample_stgy = resample_gdpprf_stgy

    @property
    def uc_tot(self):
        """ Convenience getter to extract the total uncertainty from all the GDPProfile instances.

        Returns:
            list of DataFrame: idem to self.profiles, but with only the requested data.

        """

        return [arg.uc_tot for arg in self.profiles]

    # def plot(self, x='alt', **kwargs):
    #     """ Plot method
    #
    #     Args:
    #         x (str): parameter name for the x axis. Defaults to 'alt'.
    #         **kwargs: Arbitrary keyword arguments, to be passed down to the plotting function.
    #
    #     Returns:
    #         None
    #
    #     """
    #
    #     self._plot_stgy.plot(self.profiles, self.keys, x=x, **kwargs)
