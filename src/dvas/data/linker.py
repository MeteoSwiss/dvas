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
from ..database.model import Data
from ..database.database import db_mngr

from ..config.pattern import EVENT_KEY, INSTR_KEY, ORIGMETA_KEY
from ..config.config import OrigData, OrigMeta



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

    def load(self, search):
        """

        Args:
            search (str):

        Returns:

        """

        return db_mngr.get_data(where=search)

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
        super().__init__(OUTPUT_PATH)

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

    @property
    def origdata_config_mngr(self):
        """ """
        return self._origdata_config_mngr

    def load(self, exclude_file_name=[]):
        """

        Args:
            exclude_file_name (list of str | list of Path): Already load data file to be excluded

        Returns:

        """

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
                    res = ''.join(
                        [arg[1:] for arg in
                         takewhile(lambda x: x[0] in ['#', '%'], fid)
                         ]
                    )
                else:
                    res = fid.read()

            # Read YAML config
            origmeta_cfg_mngr = OrigMeta()
            origmeta_cfg_mngr.read(res)

            print(origmeta_cfg_mngr.document)

        return

        # Create file name pattern
        manu_id_mngr = IdentifierManager(['ms', 'flight', 'batch', 'instr', 'ref_dt', 'instr_type'])
        file_nm_pat = manu_id_mngr.join(manu_id_mngr.create_id(id_source, strict=False)) + '*.csv'

        # Search file
        file_path_search = glob(str(self.repo / file_nm_pat))
        if len(file_path_search) != 1:
            raise FileNotFoundError('No file or multiple pattern')

        # Get raw data config param
        raw_cfg_param = self.raw_config_mngr.get_all(id_source)
        raw_csv_read_args = {
            key: val for key, val in raw_cfg_param.items()
            if key in PD_CSV_READ_ARGS}

        # Get parameter name
        prm_nm = self.get_item_id(id_source, 'param')

        # Modifiy names and usecols
        idx_arg_index = raw_csv_read_args['names'].index(ID_NAME)
        prm_arg_index = raw_csv_read_args['names'].index(prm_nm)
        raw_csv_read_args['usecols'] = itemgetter(idx_arg_index, prm_arg_index)(raw_csv_read_args['usecols'])
        raw_csv_read_args['names'] = itemgetter(idx_arg_index, prm_arg_index)(raw_csv_read_args['names'])

        # Read raw csv
        try:
            data = pd.read_csv(file_path_search[0], **raw_csv_read_args)
        except Exception as e:
            raise Exception(f"{id_source} / {e}")

        # Convert to dict of pd.Series
        data = data[prm_nm]

        # Apply linear correction
        data = data * raw_cfg_param[f"{prm_nm}_a"] + raw_cfg_param[f"{prm_nm}_b"]

        # Redefine index
        idx_unit = raw_cfg_param['idx_unit']
        data.index = pd.to_timedelta(data.index, idx_unit)

        return data

    def save(self, *args, **kwargs):
        """ """
        errmsg = (
            f"Save method for {self.__class__.__name__} should not be implemented."
        )
        raise NotImplementedError(errmsg)
