"""
This module contains the linker classes for data modules.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import external python packages and modules
from abc import ABC, abstractmethod
import inspect
from itertools import takewhile
import pandas as pd

# Import from current package
from ..dvas_environ import path_var as env_path_var
from ..database.model import Data, InstrType, Instrument, Tag
from ..database.database import db_mngr, EventManager
from ..config.config import OrigData, OrigMeta
from ..config.config import ConfigReadError
from ..dvas_logger import get_logger
from ..dvas_environ import glob_var as env_glob_var

# Define
INDEX_NM = Data.index.name
VALUE_NM = Data.value.name

# Pandas csv_read method arguments
PD_CSV_READ_ARGS = inspect.getfullargspec(pd.read_csv).args[1:]

# Logger
rawcsv_load = get_logger('rawcsv.load')


class DataLinker(ABC):
    """Data linker abstract class"""

    @abstractmethod
    def load(self, *args, **kwargs):
        """Data loading method"""

    @abstractmethod
    def save(self, *args, **kwargs):
        """Data saving method"""


class LocalDBLinker(DataLinker):
    """Local DB data linker """

    def load(self, search, prm_abbr):
        """Load data method

        Args:
            search (str):
            prm_abbr (str): Parameter abbr

        Returns:

        """

        # Retrieve data from DB
        data = db_mngr.get_data(where=search, prm_abbr=prm_abbr)

        # Format dataframe index
        for arg in data:
            arg['data'][INDEX_NM] = pd.TimedeltaIndex(arg['data'][INDEX_NM], 's')
            arg['data'] = arg['data'].set_index([INDEX_NM])[VALUE_NM]

        return data

    def save(self, data_list):
        """Save data method

        Args:
          data_list (list of dict): {'data': pd.Series, 'event': EventManager'}

        """

        for args in data_list:
            db_mngr.add_data(**args)


class CSVOutputLinker(DataLinker):
    """CSV output data linker class"""

    def load(self):
        errmsg = (
            f"Save method for {self.__class__.__name__} is not implemented. " +
            "No load of standardized CSV file data is implemented."
        )
        raise NotImplementedError(errmsg)

    def save(self, data):
        """

        Args:
          data:

        Returns:


        """

        # Create path
        env_path_var.output_path.mkdir(mode=777, parents=True, exist_ok=False)

        # Save data
        data.to_csv(env_path_var.output_path / 'data.csv', header=False, index=True)


class OriginalCSVLinker(DataLinker):
    """Original data CSV linger"""

    def __init__(self):
        """Constructor"""
        # Set attributes
        self._origdata_config_mngr = OrigData()

        # Init origdata config manager
        self.origdata_config_mngr.read()

    @property
    def origdata_config_mngr(self):
        """config.OrigData: Config orig data manager"""
        return self._origdata_config_mngr

    def load(self, prm_abbr, exclude_file_name=None):
        """Overwrite load method

        Args:
            prm_abbr (str): Parameter abbr
            exclude_file_name (list of str): Already load data file name to be
                excluded.

        Returns:
            list of dict

        """

        # Init
        out = []
        if exclude_file_name is None:
            exclude_file_name = []

        # Define
        cfg_file_suffix = ['.' + arg for arg in env_glob_var.config_file_ext]
        origmeta_cfg_mngr = OrigMeta()

        # Scan recursively CSV files in directory
        origdata_file_path_list = [
            arg for arg in env_path_var.orig_data_path.rglob("*.csv")
            if arg.name not in exclude_file_name
        ]

        for origdata_file_path in origdata_file_path_list:

            # Define metadata path
            try:
                metadata_file_path = next(
                    arg for arg in origdata_file_path.parent.glob(
                        '*' + origdata_file_path.stem + '*.*'
                    ) if arg.suffix in cfg_file_suffix
                )
            except StopIteration:
                metadata_file_path = origdata_file_path

            # Read metadata
            with metadata_file_path.open(mode='r') as fid:
                if metadata_file_path.suffix == '.csv':
                    meta_raw = ''.join(
                        [arg[1:] for arg in
                         takewhile(lambda x: x[0] in ['#', '%'], fid)
                         ]
                    )
                else:
                    meta_raw = fid.read()

            # Read YAML config
            try:
                origmeta_cfg_mngr.read(meta_raw)
                assert origmeta_cfg_mngr.document

            except ConfigReadError as exc:
                rawcsv_load.error(
                    "Error in reading file '%s' (%s)",
                    metadata_file_path,
                    exc
                )
                continue

            except AssertionError:
                rawcsv_load.error(
                    "No meta data found in file '%s'",
                    metadata_file_path
                )
                continue

            # Create event
            event = EventManager(
                event_dt=origmeta_cfg_mngr['event_dt'],
                instr_id=origmeta_cfg_mngr['instr_id'],
                prm_abbr=prm_abbr,
                tag_abbr=origmeta_cfg_mngr['tag_abbr'],
            )

            # Check instr_type name existence
            # (need it for loading origdata config)
            instr_type_name = db_mngr.get_or_none(
                Instrument,
                search={
                    'join_order': [InstrType],
                    'where': Instrument.instr_id == event.instr_id
                },
                attr=[[Instrument.instr_type.name, InstrType.type_name.name]]
            )

            if not instr_type_name:
                rawcsv_load.error(
                    "Missing instrument id '%s' in DB while reading " +
                    "meta data in file '%s'",
                    event.instr_id,
                    metadata_file_path
                )
                continue

            # Get origdata config params
            instr_type_name = instr_type_name[0]
            origdata_cfg_prm = self.origdata_config_mngr.get_all(
                [instr_type_name, prm_abbr, event.instr_id]
            )

            # Get raw data config param
            raw_csv_read_args = {
                key: val for key, val in origdata_cfg_prm.items()
                if key in PD_CSV_READ_ARGS}

            # Read raw csv
            try:
                data = pd.read_csv(origdata_file_path, **raw_csv_read_args)

            except ValueError as exc:
                rawcsv_load.error(
                    "Error while reading '%s' in CSV file '%s' (%s: %s)",
                    prm_abbr, origdata_file_path,
                    type(exc).__name__, exc
                )

            else:

                # Log
                rawcsv_load.info(
                    "Successful reading of '%s' in CSV file '%s'",
                    prm_abbr, origdata_file_path,
                )

                # Append data
                out.append(
                    {
                        'event': event,
                        'data': data,
                        'source_info': origdata_file_path.name
                    }
                )

        return out

    def save(self, *args, **kwargs):
        """Implement save method"""
        errmsg = (
            f"Save method for {self.__class__.__name__} should not implemented."
        )
        raise NotImplementedError(errmsg)


class ConfigInstrIdError(Exception):
    """Error for missing instrument id"""
