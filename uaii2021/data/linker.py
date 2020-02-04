
from abc import ABC, abstractmethod
from pathlib import Path
import inspect
from operator import itemgetter
from glob import glob

import pandas as pd

from ..config import ID_NAME
from ..config.config import IdentifierManager
from ..config.config import RawData


PD_CSV_READ_ARGS = inspect.getfullargspec(pd.read_csv).args[1:]


class DataLinker(ABC):

    def __init__(self):
        self._id_mngr = IdentifierManager(['ms', 'param', 'flight', 'batch', 'instr'])

    @property
    def id_mngr(self):
        return self._id_mngr

    def create_id(self, id_source):
        return self.id_mngr.create_id(id_source)

    def get_item_id(self, id_source, item_key):
        return self.id_mngr.get_item_id(id_source, item_key)

    @abstractmethod
    def load(self, id_source):
        """Data loading method"""
        pass

    @abstractmethod
    def save(self, id_source, data):
        """Data saving method"""
        pass


class LocalDBLinker(DataLinker):

    def __init__(self):
        super().__init__()

    def load(self, id_source):
        pass

    def save(self, id_source, data):
        pass


class CSVLinker(DataLinker):
    """

    """

    def __init__(self, repo):
        """

        Parameters
        ----------
        repo: pathlib.Path | str
            CSV repository path. Path must exists
        """

        super().__init__()

        # Convert to path
        repo = repo if isinstance(repo, Path) else Path(repo)

        # Test
        assert repo.exists(), f'{repo} does not exist'

        # Set attributes
        self._repo = repo

    @property
    def repo(self):
        return self._repo

    def get_file_path(self, id_source):
        identifier = self.create_id(id_source)
        filename = self._id_mngr.join(identifier) + '.csv'
        path = self.repo / Path('/'.join(self._id_mngr.split(identifier)))
        return path, filename

    def load(self, id_source):
        """

        Parameters
        ----------
        id_source: list of str | str

        Returns
        -------
        pd.Series
            Return None if file is missing

        """
        # Define
        path, filename = self.get_file_path(id_source)

        # Load
        try:
            data = pd.read_csv(path / filename, header=None, index_col=0, squeeze=True)
            data.name = None
            data.indax.name = None
            return data
        except FileNotFoundError:
            return

    def save(self, id_source, data):
        """

        Parameters
        ----------
        id_source: list of str | str
        data: pd.Series

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


class ManufacturerCSVLinker(CSVLinker):

    def __init__(self, repo, raw_config_mngr):
        """

        Parameters
        ----------
        repo: pathlib.Path | str
        raw_config_mngr: RawData
        """
        super().__init__(repo)
        self._raw_config_mngr = raw_config_mngr

    @property
    def raw_config_mngr(self):
        return self._raw_config_mngr

    def load(self, id_source):
        """

        Parameters
        ----------
        id_source

        Returns
        -------

        """

        # Create file name pattern
        manu_id_mngr = IdentifierManager(['ms', 'flight', 'batch', 'instr', 'ref_dt', 'type'])
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

        # Redefine index
        idx_unit = raw_cfg_param['idx_unit']
        data.index = pd.to_timedelta(data.index, idx_unit)

        return data

    def save(self, id_source, data):
        errmsg = (
            f"Save method for {self.__class__.__name__} is not implemented. " +
            f"No overwrite of original data"
        )
        raise NotImplementedError(errmsg)
