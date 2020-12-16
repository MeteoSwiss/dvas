"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

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

from .strategy.resample import ResampleRSDataStrategy

from .strategy.sort import SortProfileStrategy

from .strategy.sync import TimeSynchronizeStrategy

from .strategy.plot import PlotStrategy, RSPlotStrategy, GDPPlotStrategy

from .strategy.save import SaveDataStrategy

from ..database.database import OneDimArrayConfigLinker
from ..helper import RequiredAttrMetaClass
from ..helper import deepcopy

from ..errors import dvasError, DBIOError

from ..config.definitions.tag import TAG_RAW_VAL, TAG_DERIVED_VAL

# Define
FLAG = 'flag'
VALUE = 'value'
cfg_linker = OneDimArrayConfigLinker()

# Loading strategies
load_prf_stgy = LoadProfileStrategy()
load_rsprf_stgy = LoadRSProfileStrategy()
load_gdpprf_stgy = LoadGDPProfileStrategy()

# Plotting strategies
plt_prf_stgy = PlotStrategy()
plt_rsprf_stgy = RSPlotStrategy()
plt_gdpprf_stgy = GDPPlotStrategy()

sort_prf_stgy = SortProfileStrategy()
rspl_rs_stgy = ResampleRSDataStrategy()
sync_time_stgy = TimeSynchronizeStrategy()

save_prf_stgy = SaveDataStrategy()


# TODO
#  Create a factory (in terms of design patterns) to easily build MultiProfiles
class MutliProfileAbstract(metaclass=RequiredAttrMetaClass):
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
    def __init__(self, load_stgy=None, sort_stgy=None, save_stgy=None, plot_stgy=None):

        # Init attributes
        self._load_stgy = load_stgy
        self._sort_stgy = sort_stgy
        self._save_stgy = save_stgy
        self._plot_stgy = plot_stgy

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
    def info(self):
        """list of ProfileManger info: Data info"""
        return [arg.info for arg in self.profiles]

    @deepcopy
    def rm_info_tag(self, val, inplace=True):
        """Remove tag from all info tags

        Args:
            val (str|list of str): Tag values to remove
            inplace (bool, optional): Modify in place. Defaults to True.

        """
        for i in range(len(self)):
            self.profiles[i].info.rm_tag(val)

    @deepcopy
    def add_info_tag(self, val, inplace=True):
        """Add tag from all info tags

        Args:
            val (str|list of str): Tag values to add.
            inplace (bool, optional): Modify in place. Defaults to True.

        """
        for i in range(len(self)):
            self.profiles[i].info.add_tag(val)

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
            inplace (bool): Modify in place. Defaults to True.
            **kwargs: key word arguments

        Returns:
            MultiProfile: only if inplace=False

        """

        # Call the appropriate Data strategy
        data, db_df_keys = self._load_stgy.load(*args, **kwargs)

        # Test data len
        if not data:
            raise DBIOError('Load empty data')

        # Update
        self.update(db_df_keys, data)

    @deepcopy
    def sort(self, inplace=False):
        """Sort method

        Args:
            inplace (bool, `optional`): If True, perform operation in-place
                Defaults to True.
        Returns
            MultiProfileManager if inplace is True, otherwise None

        """

        # Sort
        data = self._sort_stgy.sort(self.profiles)

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

        # Add tag DERIVED
        add_tags = [TAG_DERIVED_VAL] if add_tags is None else add_tags + [TAG_DERIVED_VAL]

        # Add tags
        obj.add_info_tag(add_tags)

        # Remove tag RAW
        rm_tags = [TAG_RAW_VAL] if rm_tags is None else rm_tags + [TAG_RAW_VAL]

        # Remove tags
        obj.rm_info_tag(rm_tags)

        # Restructure the parameters into a dict, to be consistent with the rest of the class.
        if prms is None:
            prms = list(self.db_variables.keys())

        # Call save strategy
        self._save_stgy.save(obj, prms)

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
        """ Convenience getter to extract one specific parameter from the DataFrames of all the
        Profile instances.

        Args:
            prm_list (list of str): names of the parameter to extract from all the Profile
                DataFrame. Defaults to None (=return all the data from the DataFrame)

        Returns:
            dict of list of DataFrame: idem to self.profiles, but with only the requested data.

        """

        if prm_list is None:
            prm_list = self.db_variables.keys()
            #TODO: Here I need to remove all the keys that are indices.

        if isinstance(prm_list, str):
            # Assume the user forgot to put the key into a list.
            prm_list = [prm_list]

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
            raise dvasError(f"Unknown parameter/attribute name in {prm_list}")

        return out

    def get_info(self, prm):
        """ Convenience function to extract specific (a unique!) Info from all the
        Profile instances.

        Args:
            prm (str): parameter name (unique!) to extract.

        Returns:
            dict of list: idem to self.profiles, but with only the requested metadata.

        """

        return [info[prm] for info in self.info]

    def plot(self, **kwargs):
        """ Plot method

        Args:
            **kwargs: Keyword arguments to be passed down to the plotting function.

        Returns:
            None

        """

        self._plot_stgy.plot(self, **kwargs)


class MultiProfile(MutliProfileAbstract):
    """Multi profile base class, designed to handle multiple Profile."""

    #: type: supported Profile Types
    _DATA_TYPES = Profile

    def __init__(self):
        super().__init__(
            load_stgy=load_prf_stgy, sort_stgy=sort_prf_stgy,
            save_stgy=save_prf_stgy, plot_stgy=plt_prf_stgy,
        )

        # Init strategy
        #self._sync_stgy = sync_stgy

class MultiRSProfileAbstract(MutliProfileAbstract):
    """Abstract MultiRSProfile class"""

    @abstractmethod
    def __init__(
            self, load_stgy=None, sort_stgy=None,
            save_stgy=None, plot_stgy=None,
    ):
        super().__init__(
            load_stgy=load_stgy, sort_stgy=sort_stgy,
            save_stgy=save_stgy, plot_stgy=plt_prf_stgy,
        )

        # Set attributes
        #self._rspl_stgy = rspl_stgy

    # TODO
    #  Adapt for MultiIndex
    # def resample(self, *args, inplace=True, **kwargs):
    #     """Resample method
    #
    #     Args:
    #         *args: Variable length argument list.
    #         inplace (bool, `optional`): If True, perform operation in-place.
    #             Default to False.
    #         **kwargs: Arbitrary keyword arguments.
    #
    #     Returns:
    #         MultiProfileManager if inplace is True, otherwise None
    #
    #     """
    #
    #     # Resample
    #     out = self._rspl_stgy.resample(self.copy().profiles, *args, **kwargs)
    #
    #     # Load
    #     res = self.update(self.db_variables, out, inplace=inplace)
    #
    #     return res


class MultiRSProfile(MultiRSProfileAbstract):
    """Multi RS profile manager, designed to handle multiple RSProfile instances."""

    _DATA_TYPES = RSProfile

    def __init__(self):
        super().__init__(
            load_stgy=load_rsprf_stgy, sort_stgy=sort_prf_stgy,
            save_stgy=save_prf_stgy, plot_stgy=plt_prf_stgy,
        )

    #def synchronize(self, *args, inplace=False, **kwargs):
    #    """Synchronize method
    #
    #    Args:
    #        *args: Variable length argument list.
    #        inplace (bool, `optional`): If True, perform operation in-place.
    #            Default to False.
    #        **kwargs: Arbitrary keyword arguments.
    #
    #    Returns
    #        MultiProfileManager if inplace is True, otherwise None
    #
    #    """
    #
    #    # Synchronize
    #    out = self._sync_stgy.synchronize(self.copy().data, 'data', *args, **kwargs)
    #
    #    # Load
    #    res = self.load(out, inplace=inplace)
    #
    #    return res


class MultiGDPProfile(MultiRSProfileAbstract):
    """Multi GDP profile manager, designed to handle multiple GDPProfile instances."""

    _DATA_TYPES = GDPProfile

    def __init__(self):
        super().__init__(
            load_stgy=load_gdpprf_stgy, sort_stgy=sort_prf_stgy,
            save_stgy=save_prf_stgy, plot_stgy=plt_prf_stgy,
        )

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

    #def synchronize(self, *args, inplace=False, method='time', **kwargs):
    #    """Overwrite of synchronize method
    #
    #    Args:
    #        *args: Variable length argument list.
    #        inplace (bool, `optional`): If True, perform operation in-place.
    #            Default to False.
    #        method (str, `optional`): Method used to synchronize series.
    #            Default to 'time'
    #            - 'time': Synchronize on time
    #            - 'alt': Synchronize on altitude
    #        **kwargs: Arbitrary keyword arguments.
    #
    #    Returns
    #        MultiProfileManager if inplace is True, otherwise None
    #
    #    """
    #
    #    # Synchronize
    #    if method == 'time':
    #        out = self._sync_stgy.synchronize(self.copy().data, 'data', *args, **kwargs)
    #    else:
    #        #TODO modify to integrate linear AND offset compesation
    #        out = self._sync_stgy.synchronize(self.copy().data, 'alt', *args, **kwargs)
    #
    #    # Modify inplace or not
    #    if inplace:
    #        self.data = out
    #        res = None
    #    else:
    #        res = self.load(out)
    #
    #    return res
