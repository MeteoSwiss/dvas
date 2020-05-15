"""
This module contains the linker classes for data modules.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import external python packages and modules
from abc import ABC, abstractmethod
from pathlib import Path
import inspect
from itertools import takewhile
import pandas as pd

# Import from current package
from ..dvas_environ import path_var as env_path_var
from ..database.model import Data
from ..database.database import db_mngr, EventManager, InstrType, Instrument
from ..config.config import OrigData, OrigMeta
from ..dvas_helper import TimeIt

# Define
INDEX_NM = Data.index.name
VALUE_NM = Data.value.name

# Pandas csv_read method arguments
PD_CSV_READ_ARGS = inspect.getfullargspec(pd.read_csv).args[1:]


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
        "Constructor"
        # Set attributes
        self._origdata_config_mngr = OrigData()

        # Init origdata config manager
        self.origdata_config_mngr.read()

    @property
    def origdata_config_mngr(self):
        """config.OrigData: Config orig data manager"""
        return self._origdata_config_mngr

    @TimeIt()
    def load(self, prm_abbr, exclude_file_name=None):
        """Overwrite load method

        Args:
            exclude_file_name (list of str | list of Path): Already load data file to be excluded

        Returns:

        """

        # Init
        out = []
        if exclude_file_name is None:
            exclude_file_name = []

        # Define
        origmeta_cfg_mngr = OrigMeta()

        # Convert
        exclude_file_name = [Path(arg) for arg in exclude_file_name]

        # Scan recursively CSV files in directory
        origdata_file_path_list = [
            arg for arg in env_path_var.orig_data_path.rglob("*.csv")
            if arg not in exclude_file_name
        ]

        for origdata_file_path in origdata_file_path_list:

            # Define metadata path
            metadata_file_path = origdata_file_path
            for ext in ['.txt', '.yml']:
                try:
                    new_fn = next(
                        Path(metadata_file_path.parent).glob(
                            '*' + metadata_file_path.stem + '*' + ext
                        )
                    )
                except StopIteration:
                    pass
                else:
                    metadata_file_path = new_fn
                    break

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
            origmeta_cfg_mngr.read(meta_raw)

            # Create event
            event = EventManager(
                event_dt=origmeta_cfg_mngr.document['event_dt'],
                instr_id=origmeta_cfg_mngr.document['instr_id'],
                prm_abbr=prm_abbr,
                batch_id=origmeta_cfg_mngr.document['batch_id'],
                day_event=origmeta_cfg_mngr.document['day_event'],
                event_id=origmeta_cfg_mngr.document['event_id']
            )

            # Get instr_type name
            instr_type_name = db_mngr.get_or_none(
                Instrument,
                search={
                    'join_order': [InstrType],
                    'where': Instrument.instr_id == event.instr_id
                },
                attr=['instr_type', 'type_name']
            )

            if not instr_type_name:
                raise Exception('Missing instr_id')

            # Get origdata config params
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
            except Exception as _:
                raise Exception(
                    f"Error while reading CSV data in {origdata_file_path}"
                )

            out.append({'event': event, 'data': data})

        return out

    def save(self, *args, **kwargs):
        """Implement save method"""
        errmsg = (
            f"Save method for {self.__class__.__name__} should not implemented."
        )
        raise NotImplementedError(errmsg)
