"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data management

"""

# Import from external packages
from abc import abstractmethod
from copy import deepcopy
#from pampy.pampy import match, List

# Import from current package
from .linker import LocalDBLinker, CSVHandler, GDPHandler
from .strategy.data import Profile, RSProfile, GDPProfile

from .strategy.load import LoadProfileStrategy, LoadRSProfileStrategy, LoadGDPProfileStrategy

from .strategy.resample import ResampleRSDataStrategy

from .strategy.sort import TimeSortDataStrategy

from .strategy.sync import TimeSynchronizeStrategy

from .strategy.plot import PlotStrategy, RSPlotStrategy, GDPPlotStrategy

from .strategy.save import SaveDataStrategy

from ..database.database import DatabaseManager
from ..database.model import Parameter
from ..database.database import OneDimArrayConfigLinker
from ..dvas_logger import localdb, rawcsv
from ..dvas_environ import path_var
from ..dvas_helper import RequiredAttrMetaClass

from ..dvas_logger import dvasError


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

sort_time_stgy = TimeSortDataStrategy()
rspl_time_rs_stgy = ResampleRSDataStrategy()
sync_time_stgy = TimeSynchronizeStrategy()

save_prf_stgy = SaveDataStrategy()


def update_db(search, strict=False):
    """Update database.

    Args:
        search (str): prm_abbr search criteria.
        strict (bool, optional): If False, match for any sub-string.
            If True match for entire string. Default to False.

    .. uml::

        @startuml
        hide footbox

        update_db -> CSVHandler: handle(file_path, prm_abbr)
        activate CSVHandler

        CSVHandler -> GDPHandler: handle(file_path, prm_abbr)
        activate GDPHandler

        CSVHandler <- GDPHandler: data
        deactivate  GDPHandler

        update_db <- CSVHandler: data
        deactivate   CSVHandler

        @enduml


    """

    # Init linkers
    db_mngr = DatabaseManager()
    db_linker = LocalDBLinker()

    # Define chain of responsibility for loadgin from raw
    handler = CSVHandler()
    handler.set_next(GDPHandler())

    # Search prm_abbr
    if strict is True:
        search_dict = {'where': Parameter.prm_abbr == search}
    else:
        search_dict = {'where': Parameter.prm_abbr.contains(search)}

    prm_abbr_list = [
        arg[0] for arg in db_mngr.get_or_none(
            Parameter,
            search=search_dict,
            attr=[[Parameter.prm_abbr.name]],
            get_first=False
        )
    ]

    # If no matching parameters were found, issue a warning and stop here.
    if len(prm_abbr_list) == 0:
        localdb.info("No database parameter found for the query: %s", search)
        return None

    # Log
    localdb.info("Update db for following parameters: %s", prm_abbr_list)

    # Scan path
    origdata_path_scan = list(path_var.orig_data_path.rglob("*.*"))

    # Loop loading
    for prm_abbr in prm_abbr_list:

        # Log
        rawcsv.info("Start reading CSV files for '%s'", prm_abbr)

        # Scan files
        new_orig_data = []
        for file_path in origdata_path_scan:
            result = handler.handle(file_path, prm_abbr)
            if result:
                new_orig_data.append(result)

                # Log
                rawcsv.info(
                    "CSV files '%s' was treated", file_path
                )
            else:

                # Log
                rawcsv.debug(
                    "CSV files '%s' was left untouched", file_path
                )

        # Log
        rawcsv.info("Finish reading CSV files for '%s'", prm_abbr)
        rawcsv.info(
            "Found %d new data while reading CSV files for '%s'",
            len(new_orig_data),
            prm_abbr
        )

        # Log
        localdb.info(
            "Start inserting in local DB new found data for '%s'", prm_abbr
        )

        # Save to DB
        db_linker.save(new_orig_data)

        # Log
        localdb.info(
            "Finish inserting in local DB new found data for '%s'", prm_abbr
        )


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

    @abstractmethod
    def __init__(self, load_stgy=None):

        self._load_stgy = load_stgy

        # Init attributes
        self._db_variables = {}
        self._profiles = []

    @property
    def profiles(self):
        """list of Profile"""
        return self._profiles

    @profiles.setter
    def profiles(self, val):

        # Check input type
        assert isinstance(val, list), "Was expecting a list, not: %s" % (type(val))

        # Check input value type
        assert all(
            [isinstance(arg, self._DATA_TYPES) for arg in val]
        ), f"Invalid value type: {[type(arg) for arg in val]}"

        self._profiles = val

    @property
    def keys(self):
        """list: Profiles keys"""
        return self._DATA_TYPES.keys()

    @property
    def events(self):
        """list of ProfileManger event: Event metadata"""
        return [arg.event for arg in self.profiles]

    def copy(self):
        """Return a deep copy of the object"""
        return deepcopy(self)

    def load(self, *args, **kwargs):
        """Load data.

        Args:
            *args: positional arguments
            **kwargs: key word arguments

        Returns:
            MultiProfile: only if inplace=False

        """

        # Call the appropriate Data strategy
        data, db_df_keys = self._load_stgy.load(*args, **kwargs)

        self._db_variables = db_df_keys
        self.profiles = data


class MultiProfile(MutliProfileAbstract):
    """Multi profile base class, designed to handle multiple Profile."""

    #: type: supported Profile Types
    _DATA_TYPES = Profile

    def __init__(
            self,
            load_stgy=load_prf_stgy,
            #sort_stgy=sort_time_stgy,
            #plot_stgy=plt_prf_stgy, save_stgy=save_prf_stgy
    ):
        super().__init__(load_stgy=load_stgy)

        # Init strategy
        #self._resample_stgy = resample_stgy
        #self._sort_stgy = sort_stgy
        #self._sync_stgy = sync_stgy
        #self._plot_stgy = plot_stgy
        #self._save_stgy = save_stgy

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
            return {key: [arg.data for arg in item] for key, item in self.profiles.items()}

        if isinstance(prm_list, str):
            # Assume the user forgot to put the key into a list.
            prm_list = [prm_list]

        # TODO: Here, I am directly calling the DataFrame columns ...
        # But how could I also allow to call "convenience functions" like uc_tot in GDPProfile ?

        # Check that the parameters are valid and exist
        for key in self.DB_VARIABLES:
            assert all([prm in self.DB_VARIABLES[key] for prm in prm_list]), \
                "Unknown parameter name. Should be one of %s" % (self.DB_VARIABLES[key].keys())

        return {key: [arg.data[prm_list] for arg in item] for key, item in self.profiles.items()}

    def get_evt_prm(self, prm):
        """ Convenience function to extract specific (a unique!) Event metadata from all the
        Profile instances.

        Args:
            prm (str): parameter name (unique!) to extract from all the Events.

        Returns:
            dict of list: idem to self.profiles, but with only the requested metadata.

        """

        return {key: [evt.as_dict()[prm] for evt in item] for key, item in self.events.items()}

    #TODO: add an "append" method to add new data manually

    def sort(self, inplace=False):
        """Sort method

        Args:
            inplace (bool, `optional`): If True, perform operation in-place
                Default to False.
        Returns
            MultiProfileManager if inplace is True, otherwise None

        """

        # Sort
        out = self._sort_stgy.sort(self.copy().data)

        # Load
        res = self.load(out, inplace=inplace)

        return res

    def plot(self, **kwargs):
        """ Plot method

        Args:
            **kwargs: Arbitrary keyword arguments, to be passed down to the plotting function.

        Returns:
            None

        """

        self._plot_stgy.plot(self.profiles, self.keys, **kwargs)

    def save(self, add_tags=None, rm_tags=None, prm_list=None):
        """Save method to store the *entire* content of the Multiprofile instance back into the
        database with an updated set of tags.

        Args:
            add_tags (list of str): list of tags to add to event.
            rm_tags (list of str, optional): list of *existing* tags to remove from the event.
                Defaults to None.
            prms (list of str, optional): list of column names to save to the database.
                Defaults to None (= save all possible parameters).

        """

        if add_tags is None and rm_tags is None:
            raise dvasError('Need either add_tags or rm_tags to be specified.')
        # Restructure the parameters into a dict, to be consistent with the rest of the class.
        #if prm_list is None:
        #    prms = {item:self.DB_VARIABLES[item].keys() for item in self.DB_VARIABLES}
        #else:
        #    prms = {item:prms for item in self.DB_VARIABLES}

        # Call save strategy
        self._save_stgy.save(self.get_prms(prm_list), self.copy().events,
                             self.DB_VARIABLES, add_tags, rm_tags)

    # TODO: implement an "export" function that can export specific DataFrame columns back into
    # the database under new variable names ?


class MultiRSProfile(MultiProfile):
    """Multi RS profile manager, designed to handle multiple RSProfile instances."""

    _DATA_TYPES = RSProfile

    def __init__(self):
        super().__init__(load_stgy=load_rsprf_stgy)

    #def resample(self, *args, inplace=False, **kwargs):
    #    """Resample method
    #
    #    Args:
    #        *args: Variable length argument list.
    #        inplace (bool, `optional`): If True, perform operation in-place.
    #            Default to False.
    #        **kwargs: Arbitrary keyword arguments.
    #
    #    Returns:
    #        MultiProfileManager if inplace is True, otherwise None
    #
    #    """
    #
    #    # Resample
    #    out = self._resample_stgy.resample(self.copy().data, *args, **kwargs)
    #
    #    # Load
    #    res = self.load(out, inplace=inplace)
    #
    #    return res

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


class MultiGDPProfile(MultiProfile):
    """Multi GDP profile manager, designed to handle multiple GDPProfile instances."""

    _DATA_TYPES = GDPProfile

    def __init__(self):
        super().__init__(load_stgy=load_gdpprf_stgy)

    @property
    def uc_tot(self):
        """ Convenience getter to extract the total uncertainty from all the GDPProfile instances.

        Returns:
            list of DataFrame: idem to self.profiles, but with only the requested data.

        """

        return [arg.uc_tot for arg in self.profiles]

    def plot(self, x='alt', **kwargs):
        """ Plot method

        Args:
            x (str): parameter name for the x axis. Defaults to 'alt'.
            **kwargs: Arbitrary keyword arguments, to be passed down to the plotting function.

        Returns:
            None

        """

        self._plot_stgy.plot(self.profiles, self.keys, x=x, **kwargs)

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
