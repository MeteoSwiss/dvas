"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data management

"""

# Import from external packages
#from abc import abstractmethod
from copy import deepcopy
#from pampy.pampy import match, List

# Import from current package
from .linker import LocalDBLinker, CSVHandler, GDPHandler
from .strategy.data import Profile, RSProfile, GDPProfile

from .strategy.load import LoadProfileStrategy, LoadRSProfileStrategy, LoadGDPProfileStrategy

from .strategy.resample import ResampleRSDataStrategy

from .strategy.sort import TimeSortDataStrategy

from .strategy.sync import TimeSynchronizeStrategy

from .strategy.plot import TimePlotStrategy
from .strategy.plot import AltTimePlotStartegy

from .strategy.save import SaveTimeDataStrategy

from ..database.database import DatabaseManager
from ..database.model import Parameter
from ..database.database import OneDimArrayConfigLinker
from ..dvas_logger import localdb, rawcsv
from ..dvas_environ import path_var
from ..dvas_helper import RequiredAttrMetaClass


# Define
FLAG = 'flag'
VALUE = 'value'
cfg_linker = OneDimArrayConfigLinker()

load_prf_stgy = LoadProfileStrategy()
load_rsprf_stgy = LoadRSProfileStrategy()
load_gdpprf_stgy = LoadGDPProfileStrategy()

sort_time_stgy = TimeSortDataStrategy()
rspl_time_rs_stgy = ResampleRSDataStrategy()
sync_time_stgy = TimeSynchronizeStrategy()
plt_time_stgy = TimePlotStrategy()
plt_alt_time_stgy = AltTimePlotStartegy()
save_time_stgy = SaveTimeDataStrategy()


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
        search = {'where': Parameter.prm_abbr == search}
    else:
        search = {'where': Parameter.prm_abbr.contains(search)}

    prm_abbr_list = [
        arg[0] for arg in db_mngr.get_or_none(
            Parameter,
            search=search,
            attr=[[Parameter.prm_abbr.name]],
            get_first=False
        )
    ]

    # Log
    localdb.info(
        "Update db for following parameters: %s",
        prm_abbr_list
    )

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


class MultiProfile(metaclass=RequiredAttrMetaClass):
    """Multi profile base class, designed to handle multiple Profile."""

    REQUIRED_ATTRIBUTES = {'_DATA_TYPES': dict}

    #: dict: list of supported Profile Types
    _DATA_TYPES = {'prf': Profile}

    def __init__(self,
                 load_stgy=load_prf_stgy, sort_stgy=sort_time_stgy, plot_stgy=plt_time_stgy,
                 save_stgy=save_time_stgy):
        """ Init function, setting up the applicable strategies. """

        # Init strategy
        self._load_stgy = load_stgy
        #self._resample_stgy = resample_stgy
        self._sort_stgy = sort_stgy
        #self._sync_stgy = sync_stgy
        self._plot_stgy = plot_stgy
        self._save_stgy = save_stgy

        # Init attributes
        #TODO: do we really need to keep this as a dictionnary ? In case of a mix ?
        self._profiles = {}
        for key in self._DATA_TYPES:
            self._profiles[key] = []

    @property
    def keys(self):
        """list: Data types"""
        return self._DATA_TYPES.keys()

    @property
    def profiles(self):
        """list of Profile"""
        return self._profiles

    @profiles.setter
    def profiles(self, val):

        assert isinstance(val, dict), "Was expecting a dict, not: %s" % (type(val))
        assert self._DATA_TYPES.keys() == val.keys(), "Invalid keys: %s" % (val.keys())

        #TODO: check that all the profiles have the correct type before settign them.

        self._profiles = val

    def get_prms(self, prm_list=':'):
        """ Convenience getter to extract one specific parameter from the DataFrames of all the
        Profile instances.

        Args:
            prm_list (list of str): names of the parameter to extract from all the Profile
                DataFrame. Defaults to ':' (=return all the data from the DataFrame)

        Returns:
            dict of list of DataFrame: idem to self.profiles, but with only the requested data.

        """

        # TODO: Here, I am directly calling the DataFrame columns ...
        # But how could I also allow to call "convenience functions" like uc_tot in GDPProfile ?
        return {key: [arg.data[prm_list] for arg in item] for key, item in self.profiles.items()}

    @property
    def events(self):
        """dict of list of ProfileManger event: Event metadata"""
        return {key: [arg.event for arg in val] for key, val in self.profiles.items()}

    def get_evt_prm(self, prm):
        """ Convenience function to extract specific (a unique!) Event metadata from all the
        Profile instances.

        Args:
            prm (str): parameter name (unique!) to extract from all the Events.

        Returns:
            dict of list: idem to self.profiles, but with only the requested metadata.

        """

        return {key: [evt.as_dict()[prm] for evt in item] for key, item in self.events.items()}

    # TODO: why do we need this ?
    def __getitem__(self, item):
        return self.profiles[item]

    def copy(self):
        """Retrun a deep copy of the object"""
        return deepcopy(self)

    #TODO: add an "append" method to add new data manually

    def load(self, *args, inplace=False, **kwargs):
        """Load classmethod

        Args:
            inplace (bool, `optional`): If True, perform operation in-place.
                Default to False.

        Returns:
            MultiProfile: only if inplace=False

        """

        # vof: removing this in favor of a dedicated "append" method.
        # Load data
        #if len(args) == 1:
        #    data = args[0]
        #else:

        # Call the appropriate Data Startegy
        data = self._load_stgy.load(*args, **kwargs)

        # Modify inplace or not
        if inplace is True:
            self.profiles = data
            res = None
        else:
            res = self.copy()
            res.profiles = data

        return res

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

    #def plot(self, *args, **kwargs):
    #    """Plot method
    #
    #    Args:
    #        *args: Variable length argument list.
    #        **kwargs: Arbitrary keyword arguments.
    #
    #    Returns:
    #        None
    #
    #    """
    #    self._plot_stgy.plot(self.values, self.event_mngrs, *args, **kwargs)

    #def save(self, prm_abbr):
    #    """Save method
    #
    #    Args:
    #        prm_abbr (dict): Parameter abbr.
    #
    #    Returns:
    #        None
    #
    #    """
    #
    #    # Test
    #    assert match(
    #        prm_abbr, {key: str for key in self._DATA_TYPES.keys()},
    #        True, default=False
    #    )
    #
    #    # Call save strategy
    #    self._save_stgy.save(self.copy().values, self.event_mngrs, prm_abbr)


class MultiRSProfile(MultiProfile):
    """Multi RS profile manager, designed to handle multiple RSProfile instances."""

    _DATA_TYPES = {'rs_prf': RSProfile}

    def __init__(self,
                 load_stgy=load_rsprf_stgy, sort_stgy=sort_time_stgy, plot_stgy=plt_time_stgy,
                 save_stgy=save_time_stgy, resample_stgy=rspl_time_rs_stgy,
                 sync_stgy=sync_time_stgy):
        """ Init function, to set the appropriate strategies"""

        # First call the parent __init__() for the common strategies.
        super().__init__(load_stgy=load_stgy, sort_stgy=sort_stgy, plot_stgy=plot_stgy,
                         save_stgy=save_stgy)

        # Then also set the specific strategies for this Class
        self._resample_stgy = resample_stgy
        self._sync_stgy = sync_stgy

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

class MultiGDPProfile(MultiRSProfile):
    """Multi GDP profile manager, designed to handle multiple GDPProfile instances."""

    _DATA_TYPES = {'gdp_prf': GDPProfile}

    def __init__(self,
                 load_stgy=load_gdpprf_stgy, sort_stgy=sort_time_stgy, plot_stgy=plt_time_stgy,
                 save_stgy=save_time_stgy, resample_stgy=rspl_time_rs_stgy,
                 sync_stgy=sync_time_stgy):
        """ Init function, to set the appropriate strategies"""

        # First call the parent __init_() to set the common strategies.
        super().__init__(load_stgy=load_stgy, sort_stgy=sort_stgy, plot_stgy=plot_stgy,
                         save_stgy=save_stgy, resample_stgy=resample_stgy, sync_stgy=sync_stgy)


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
