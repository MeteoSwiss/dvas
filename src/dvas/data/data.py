"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data management

"""

# Import from external packages
from abc import ABCMeta, abstractmethod
from operator import itemgetter
from copy import deepcopy
import numpy as np
import pandas as pd

# Import from current package
from .linker import LocalDBLinker, CSVHandler, GDPHandler
from .strategy.load import LoadTimeDataStrategy
from .strategy.load import LoadAltDataStrategy
from .strategy.resample import TimeResampleDataStrategy
from .strategy.sort import TimeSortDataStrategy
from .strategy.sync import TimeSynchronizeStrategy
from ..plot.plot import basic_plot
from .math import crosscorr
from ..database.database import db_mngr
from ..database.model import Parameter
from ..database.database import OneDimArrayConfigLinker
from ..dvas_logger import localdb, rawcsv
from ..dvas_environ import path_var


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


class MultiProfileManager(metaclass=ABCMeta):
    """Multi profile manager"""

    @abstractmethod
    def __init__(
            self,
            load_stgy, resample_stgy, sort_stgy,
            sync_stgy, plot_stgy
    ):

        # Init strategy
        self._load_stgy = load_stgy
        self._resample_stgy = resample_stgy
        self._sort_stgy = sort_stgy
        self._sync_stgy = sync_stgy
        self._plot_stgy = plot_stgy

        # Init attributes
        self._data = {}

    @property
    def data(self):
        """dict of list of ProfileManager: Data"""
        return self._data

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
        return deepcopy(self)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load method"""

        # Create instance
        inst = cls()

        # Load data
        if len(args) == 1:
            inst._data = args[0]
        else:
            inst._data = inst._load_stgy.load(*args, **kwargs)

        return inst

    def resample(self, *args, **kwargs):
        """Resample method

        Returns:
            MultiProfileManager

        """
        out = self._resample_stgy.resample(self.copy().data, *args, **kwargs)
        return self.load(out)

    def sort(self):
        """Sort method

        Returns
            None

        """
        out = self._sort_stgy.sort(self.copy().data)
        return self.load(out)

    def synchronize(self, *args, **kwargs):
        """Synchronize method

        Returns
            MultiProfileManager

        """
        out = self._sync_stgy.synchronize(self.copy().data, 'data', *args, **kwargs)
        return self.load(out)


class TemporalMultiProfileManager(MultiProfileManager):
    """Temporal multi profile manager"""

    def __init__(self):
        super().__init__(
            load_time_stgy, rspl_time_stgy, sort_time_stgy,
            sync_time_stgy, None
        )


class AltitudeMultiProfileManager(MultiProfileManager):
    """Altitude multi profile manager"""

    def __init__(self):
        super().__init__(
            load_alt_stgy, None, None,
            sync_time_stgy, None
        )


    def map(self, func, inplace, *args, **kwargs):
        """Map individual TimeProfileManager"""
        if inplace:
            MultiTimeProfileManager(
                map(lambda x: func(x, *args, **kwargs), self)
            )
            out = None
        else:
            out = self.copy()
            MultiTimeProfileManager(
                map(lambda x: func(x, *args, **kwargs), out)
            )
        return out


    def append(self, value):
        """Overwrite of append method"""
        assert isinstance(value, (TimeProfileManager, EmptyProfileManager))
        super().append(value)

    def resample(self, interval='1s', method='mean', inplace=False):
        """Resample method"""
        return self.map(
            TimeProfileManager.resample, inplace=inplace,
            interval=interval, method=method
        )

    def interpolate(self, inplace=False):
        """Interpolate method"""
        return self.map(
            TimeProfileManager.interpolate, inplace=inplace
        )


    def plot(self, **kwargs):
        """Plot method"""

        basic_plot(self, **kwargs)
