"""
Copyright(c) 2020 MeteoSwiss, contributors listed in AUTHORS

Distributed under the terms of the BSD 3 - Clause License.

SPDX - License - Identifier: BSD - 3 - Clause

Module contents: Data linker classes

"""

# Import external python packages and modules
from abc import ABC, abstractmethod
import re
from itertools import chain, zip_longest
from itertools import takewhile
import inspect
import netCDF4  as nc
import pandas as pd

# Import from current package
from ..dvas_environ import path_var as env_path_var
from ..database.model import Data, InstrType, Instrument, EventsInfo, Tag
from ..database.database import db_mngr, EventManager
from ..config.config import OrigData, OrigMeta, GDPData
from ..config.config import ConfigReadError
from ..dvas_logger import rawcsv
from ..dvas_environ import glob_var
from ..dvas_helper import get_by_path

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
        try:
            env_path_var.output_path.mkdir(parents=True, exist_ok=False)
            # Set user read/write permission
            env_path_var.output_path.chmod(
                env_path_var.output_path.stat().st_mode | 0o600
            )
        except (OSError,) as exc:
            raise OutputDirError(
                f"Error in creating '{env_path_var.output_path}' ({exc})"
            )

        # Save data
        data.to_csv(env_path_var.output_path / 'data.csv', header=False, index=True)


class OriginalCSVLinker(DataLinker):
    """Original data CSV linger"""

    def __init__(self):
        """Constructor"""
        # Set attributes
        self._origdata_config_mngr = OrigData()

        # Init origdata config manager
        self._origdata_config_mngr.read()

    def load(self, prm_abbr, exclude_file_name=None):
        """Load method

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
        cfg_file_suffix = ['.' + arg for arg in glob_var.config_file_ext]
        origmeta_cfg_mngr = OrigMeta()

        # Scan recursively CSV files in directory
        origdata_file_path_list = [
            arg for arg in env_path_var.orig_data_path.rglob("*.csv")
            if arg.name not in exclude_file_name
        ]

        # Loop over data files
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
                rawcsv.error(
                    "Error in reading file '%s' (%s)",
                    metadata_file_path,
                    exc
                )
                continue

            except AssertionError:
                rawcsv.error(
                    "No meta data found in file '%s'",
                    metadata_file_path
                )
                continue

            # Check instr_type name existence
            # (need it for loading origdata config)
            instr_type_name = db_mngr.get_or_none(
                Instrument,
                search={
                    'join_order': [InstrType],
                    'where': Instrument.sn == origmeta_cfg_mngr['sn']
                },
                attr=[[Instrument.instr_type.name, InstrType.type_name.name]]
            )

            if not instr_type_name:
                rawcsv.error(
                    "Missing instrument SN '%s' in DB while reading " +
                    "meta data in file '%s'",
                    origmeta_cfg_mngr['sn'],
                    metadata_file_path
                )
                continue

            # Create event
            event = EventManager(
                event_dt=origmeta_cfg_mngr['event_dt'],
                sn=origmeta_cfg_mngr['sn'],
                prm_abbr=prm_abbr,
                tag_abbr=origmeta_cfg_mngr['tag_abbr'],
            )

            # Get origdata config params
            instr_type_name = instr_type_name[0]
            origdata_cfg_prm = self._origdata_config_mngr.get_all(
                [instr_type_name, prm_abbr]
            )

            # Get raw data config param
            raw_csv_read_args = {
                key: val for key, val in origdata_cfg_prm.items()
                if key in PD_CSV_READ_ARGS}

            # Read raw csv
            try:
                data = pd.read_csv(origdata_file_path, **raw_csv_read_args)
                data = data.applymap(eval(origdata_cfg_prm['lambda']))

            except ValueError as exc:
                rawcsv.error(
                    "Error while reading '%s' in CSV file '%s' (%s: %s)",
                    prm_abbr, origdata_file_path,
                    type(exc).__name__, exc
                )

            else:

                # Log
                rawcsv.info(
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
            f"Save method for {self.__class__.__name__} should " +
            " not be implemented."
        )
        raise NotImplementedError(errmsg)


class GDPDataLinker(DataLinker):
    """Gruan Data Product linker"""

    def __init__(self):
        # Set attributes
        self._gdpdata_config_mngr = GDPData()

        # Init origdata config manager
        self._gdpdata_config_mngr.read()

    def load(self, prm_abbr, exclude_file_name):
        """Load method

        Args:
            prm_abbr (str): Parameter abbr
            exclude_file_name (list of str): Already load data file name to be
                excluded.

        Returns:
            list of dict

        """

        # Define
        sep = glob_var.config_gen_grp_sep
        pat_spilt = r'\{0}[^\n\r\t\{0}]+\{0}'.format(sep)
        pat_find = r'\{0}([^\n\r\t{0}]+)\{0}'.format(sep)

        # Init
        out = []
        if exclude_file_name is None:
            exclude_file_name = []

        # Scan recursively CSV files in directory
        origdata_file_path_list = [
            arg for arg in env_path_var.orig_data_path.rglob("*.nc")
            if arg.name not in exclude_file_name
        ]

        # Loop over data files
        for origdata_file_path in origdata_file_path_list:

            # Init
            meta_data = {}

            # Get instrument type from file name
            if (grp := re.search(
                    r"^[A-Z]{3}\-[A-Z]{2}\-\d{2}\_\d\_([\w\-]+\_\d{3})\_\d{8}T",
                    origdata_file_path.name
            )) is None:
                pass
                #TODO add log

            instr_type_name = grp.group(1)

            #TODO check if instrumnet exist in DB

            # Get config
            gdpdata_cfg_prm = self._gdpdata_config_mngr.get_all(
                [instr_type_name, prm_abbr]
            )

            #try:
            with nc.Dataset(origdata_file_path, 'r') as fid:

                meta_dict = {
                    EventsInfo.event_dt.name: 'dt_field',
                    Instrument.sn.name: 'sn_field',
                    Tag.tag_abbr.name: 'tag_field',
                }

                for key, value in meta_dict.items():

                    field_val = gdpdata_cfg_prm[value]

                    if isinstance(field_val, str):

                        meta_val = ''.join(
                            [arg for arg in chain(
                                *zip_longest(

                                    # Split formula
                                    re.split(pat_spilt, field_val),

                                    # Find nested item and substitute
                                    [
                                        get_by_path(fid, [item]) for item in
                                        re.findall(pat_find, field_val)
                                    ]

                                )
                            ) if arg is not None]
                        )

                    else:
                        meta_val = [
                            ''.join(
                                [arg for arg in chain(
                                    *zip_longest(

                                        # Split formula
                                        re.split(pat_spilt, field_val[i]),

                                        # Find nested item and substitute
                                        [
                                            get_by_path(fid, [item]) for item in
                                            re.findall(pat_find, field_val[i])
                                        ]

                                    )
                                ) if arg is not None]
                            ) for i in range(len(field_val))
                        ]

                    meta_data.update({key: meta_val})

                #TODO check SN -> Instr Type in DB == instr type file

                # Create event
                event = EventManager(
                    event_dt=meta_data['event_dt'],
                    sn=meta_data['sn'],
                    prm_abbr=prm_abbr,
                    tag_abbr=meta_data['tag_abbr'],
                )

                # Read data
                data_dict = {
                    Data.index.name: 'time_field',
                    Data.value.name: 'param_field',
                }

                try:
                    data = pd.DataFrame(
                        {
                            Data.index.name: fid[gdpdata_cfg_prm[data_dict[Data.index.name]]][:],
                            Data.value.name: fid[gdpdata_cfg_prm[data_dict[Data.value.name]]][:],
                        }
                    )
                    data.set_index(Data.index.name, inplace=True)
                    data = data.applymap(eval(gdpdata_cfg_prm['lambda']))

                #TODO manage exception
                except Exception as exc:
                    raise Exception(exc)

                else:

                    # TODO Add log

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
            f"Save method for {self.__class__.__name__} should" +
            " not be implemented."
        )
        raise NotImplementedError(errmsg)


class ConfigInstrIdError(Exception):
    """Error for missing instrument id"""


class OutputDirError(Exception):
    """Error for bad output directory path"""
