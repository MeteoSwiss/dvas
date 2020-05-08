"""

"""

from abc import ABC, abstractmethod
from pathlib import Path
import inspect
from operator import itemgetter
from glob import glob
from itertools import takewhile

import pandas as pd

# Import from current package
from ..dvas_environ import orig_data_path as env_orig_data_path
from ..dvas_environ import config_dir_path as env_cfg_dir_path
from ..dvas_environ import output_path as env_output_path
from ..database.model import Data
from ..database.database import db_mngr, EventManager, InstrType, Instrument

from ..config.pattern import EVENT_KEY, INSTR_KEY, ORIGMETA_KEY
from ..config.config import OrigData, OrigMeta

from ..dvas_helper import TimeIt

# Define
INDEX_NM = Data.index.name
VALUE_NM = Data.value.name

# Pandas csv_read method arguments
PD_CSV_READ_ARGS = inspect.getfullargspec(pd.read_csv).args[1:]


class DataLinker(ABC):
    """ """

    @abstractmethod
    def load(self):
        """Data loading method"""
        pass

    @abstractmethod
    def save(self):
        """Data saving method"""
        pass


class LocalDBLinker(DataLinker):
    """ """

    def __init__(self):
        super().__init__()

    def load(self, search, prm_abbr):
        """

        Args:
            search (str):

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
        """

        Args:
          data_list (list of dict): {'data': pd.Series, 'event': EventManager'}

        Returns:


        """

        for args in data_list:
            db_mngr.add_data(**args)


class CSVLinker(DataLinker):
    """ """

    def __init__(self, repo):
        """ """

        # Set attributes
        self.repo = repo

    @property
    def repo(self):
        """ """
        return self._repo

    @repo.setter
    def repo(self, value):
        # Convert to path
        value = value if isinstance(value, Path) else Path(value)

        # Test
        assert value.exists(), f'{value} does not exist'

        self._repo = value

    def load(self):
        """Data loading method"""
        raise NotImplementedError(
            f'Please implement {self.__class__.__name__}.load()')

    def save(self):
        """Data saving method"""
        raise NotImplementedError(
            f'Please implement {self.__class__.__name__}.save()')


class CSVOutputLinker(CSVLinker):
    """ """

    _INDEX_KEY_ORDER = []

    def __init__(self):
        super().__init__(env_output_path)

    def get_file_path(self, index):
        """

        Args:
          id_source:

        Returns:


        """
        identifier = self.create_id(id_source)
        filename = self._id_mngr.join(identifier) + '.csv'
        path = self.repo / Path('/'.join(self._id_mngr.split(identifier)))
        return path, filename

    def load(self):
        errmsg = (
            f"Save method for {self.__class__.__name__} is not implemented. " +
            f"No load of standardized CSV file data is implemented."
        )
        raise NotImplementedError(errmsg)

    def save(self, id_source, data):
        """

        Args:
          id_source:
          data:

        Returns:


        """

        # Define
        path, filename = self.get_file_path(id_source)

        # Create path
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass

        # Save data
        data.to_csv(path / filename, header=False, index=True)


class OriginalCSVLinker(CSVLinker):

    def __init__(self):
        super().__init__(env_orig_data_path)

        # Set attributes
        self._origdata_config_mngr = OrigData(env_cfg_dir_path)

        # Init origdata config manager
        self._origdata_config_mngr.read()

    @property
    def origdata_config_mngr(self):
        """ """
        return self._origdata_config_mngr

    @TimeIt()
    def load(self, prm_abr, exclude_file_name=[]):
        """

        Args:
            exclude_file_name (list of str | list of Path): Already load data file to be excluded

        Returns:

        """

        # Init
        out = []

        # Define
        origmeta_cfg_mngr = OrigMeta()

        # Convert
        exclude_file_name = [Path(arg) for arg in exclude_file_name]

        # Scan recursively CSV files in directory
        origdata_file_path_list = [
            arg for arg in Path(self.repo).rglob("*.csv")
            if arg not in exclude_file_name
        ]

        for i, origdata_file_path in enumerate(origdata_file_path_list):

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
                prm_abbr=prm_abr,
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
                [instr_type_name, prm_abr, event.instr_id]
            )

            # Get raw data config param
            raw_csv_read_args = {
                key: val for key, val in origdata_cfg_prm.items()
                if key in PD_CSV_READ_ARGS}

            # Read raw csv
            try:
                data = pd.read_csv(origdata_file_path, **raw_csv_read_args)
            except Exception as e:
                raise Exception(
                    f"Error while reading CSV data in {origdata_file_path}"
                )

            out.append({'event': event, 'data': data})

        return out

    def save(self, *args, **kwargs):
        """ """
        errmsg = (
            f"Save method for {self.__class__.__name__} should not be implemented."
        )
        raise NotImplementedError(errmsg)
