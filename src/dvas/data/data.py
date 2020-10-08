"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data management

"""

# Import from external packages
from abc import abstractmethod
from copy import deepcopy
from pampy.pampy import match, List

# Import from current package
from .linker import LocalDBLinker, CSVHandler, GDPHandler
from .strategy.data import TimeProfileManager
from .strategy.load import LoadTimeDataStrategy
from .strategy.load import LoadAltDataStrategy
from .strategy.resample import TimeResampleDataStrategy
from .strategy.sort import TimeSortDataStrategy
from .strategy.sync import TimeSynchronizeStrategy
from .strategy.plot import TimePlotStrategy
from .strategy.plot import AltTimePlotStartegy
from .strategy.save import SaveTimeDataStrategy
from ..database.database import db_mngr
from ..database.model import Parameter
from ..database.database import OneDimArrayConfigLinker
from ..dvas_logger import localdb, rawcsv
from ..dvas_environ import path_var
from ..dvas_helper import RequiredAttrMetaClass


# Define
FLAG = 'flag'
VALUE = 'value'
cfg_linker = OneDimArrayConfigLinker()


# Init strategy instances
load_time_stgy = LoadTimeDataStrategy()
load_alt_stgy = LoadAltDataStrategy()
sort_time_stgy = TimeSortDataStrategy()
rspl_time_stgy = TimeResampleDataStrategy()
sync_time_stgy = TimeSynchronizeStrategy()
plot_time_stgy = TimePlotStrategy()
plot_alt_time_stgy = AltTimePlotStartegy()
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


class MultiProfileManager(metaclass=RequiredAttrMetaClass):
    """Multi profile manager"""

    REQUIRED_ATTRIBUTES = {
        '_DATA_TYPES': dict
    }

    #: dict: Data type with corresponding key name
    _DATA_TYPES = None

    @abstractmethod
    def __init__(
            self,
            load_stgy, resample_stgy, sort_stgy,
            sync_stgy, plot_stgy, save_stgy,
    ):

        # Init strategy
        self._load_stgy = load_stgy
        self._resample_stgy = resample_stgy
        self._sort_stgy = sort_stgy
        self._sync_stgy = sync_stgy
        self._plot_stgy = plot_stgy
        self._save_stgy = save_stgy

        # Init attributes
        self._data = {}

    @property
    def keys(self):
        """list: Data keys"""
        return self._DATA_TYPES.keys()

    @property
    def data(self):
        """dict of list of ProfileManager: Data"""
        return self._data

    @data.setter
    def data(self, val):

        # Test
        pattern = {
            key: List[value]
            for key, value in self._DATA_TYPES.items()
        }
        assert match(
            val, pattern,
            True, default=False
        ), 'Bad type for value'

        self._data = val

    @property
    def datas(self):
        """dict of list of ProfileManger datas: Datas"""
        return {key: [arg.data for arg in val] for key, val in self.data.items()}

    @property
    def values(self):
        """dict of list of ProfileManger values: Values"""
        return {key: [arg.value for arg in val] for key, val in self.data.items()}

    @property
    def flags(self):
        """dict of list of ProfileManger flags: Flags"""
        return {key: [arg.flag for arg in val] for key, val in self.data.items()}

    @property
    def event_mngrs(self):
        """dict of list of ProfileManger event_mngr: Event managers"""
        return {key: [arg.event_mngr for arg in val] for key, val in self.data.items()}

    def __getitem__(self, item):
        return self.data[item]

    def copy(self):
        """Retrun a deep copy of the object"""
        return deepcopy(self)

    def load(self, *args, inplace=False, **kwargs):
        """Load classmethod

        Args:
            inplace (bool, `optional`): If True, perform operation in-place.
                Default to False.

        Returns:
            MultiProfileManager

        """

        # Load data
        if len(args) == 1:
            data = args[0]
        else:
            data = self._load_stgy.load(*args, **kwargs)

        # Modify inplace or not
        if inplace is True:
            self.data = data
            res = None
        else:
            res = self.copy()
            res.data = data

        return res

    def resample(self, *args, inplace=False, **kwargs):
        """Resample method

        Args:
            *args: Variable length argument list.
            inplace (bool, `optional`): If True, perform operation in-place.
                Default to False.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            MultiProfileManager if inplace is True, otherwise None

        """

        # Resample
        out = self._resample_stgy.resample(self.copy().data, *args, **kwargs)

        # Load
        res = self.load(out, inplace=inplace)

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

    def synchronize(self, *args, inplace=False, **kwargs):
        """Synchronize method

        Args:
            *args: Variable length argument list.
            inplace (bool, `optional`): If True, perform operation in-place.
                Default to False.
            **kwargs: Arbitrary keyword arguments.

        Returns
            MultiProfileManager if inplace is True, otherwise None

        """

        # Synchronize
        out = self._sync_stgy.synchronize(self.copy().data, 'data', *args, **kwargs)

        # Load
        res = self.load(out, inplace=inplace)

        return res


    def plot(self, *args, **kwargs):
        """Plot method

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        """
        self._plot_stgy.plot(self.values, self.event_mngrs, *args, **kwargs)

    def save(self, prm_abbr):
        """Save method

        Args:
            prm_abbr (dict): Parameter abbr.

        Returns:
            None

        """

        # Test
        assert match(
            prm_abbr, {key: str for key in self._DATA_TYPES.keys()},
            True, default=False
        )

        # Call save strategy
        self._save_stgy.save(self.copy().values, self.event_mngrs, prm_abbr)


class TemporalMultiProfileManager(MultiProfileManager):
    """Temporal multi profile manager"""

    _DATA_TYPES = {'data': TimeProfileManager}

    def __init__(self):
        super().__init__(
            load_time_stgy, rspl_time_stgy, sort_time_stgy,
            sync_time_stgy, plot_time_stgy, save_time_stgy
        )


class AltitudeMultiProfileManager(MultiProfileManager):
    """Altitude multi profile manager"""

    _DATA_TYPES = {'data': TimeProfileManager, 'alt': TimeProfileManager}

    def __init__(self):
        super().__init__(
            load_alt_stgy, rspl_time_stgy, sort_time_stgy,
            sync_time_stgy, plot_alt_time_stgy, save_time_stgy
        )

    def synchronize(self, *args, inplace=False, method='time', **kwargs):
        """Overwrite of synchronize method

        Args:
            *args: Variable length argument list.
            inplace (bool, `optional`): If True, perform operation in-place.
                Default to False.
            method (str, `optional`): Method used to synchronize series.
                Default to 'time'
                - 'time': Synchronize on time
                - 'alt': Synchronize on altitude
            **kwargs: Arbitrary keyword arguments.

        Returns
            MultiProfileManager if inplace is True, otherwise None

        """

        # Synchronize
        if method == 'time':
            out = self._sync_stgy.synchronize(self.copy().data, 'data', *args, **kwargs)
        else:
            #TODO modify to integrate linear AND offset compesation
            out = self._sync_stgy.synchronize(self.copy().data, 'alt', *args, **kwargs)

        # Modify inplace or not
        if inplace:
            self.data = out
            res = None
        else:
            res = self.load(out)

        return res
