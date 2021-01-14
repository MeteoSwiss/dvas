"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data linker classes

"""

# Import external python packages and modules
from abc import ABC, abstractmethod
import re
from itertools import takewhile
import inspect
import netCDF4 as nc
import pandas as pd

# Import from current package
from ..environ import path_var as env_path_var
from ..database.model import InstrType, Info
from ..database.model import Parameter, DataSource
from ..database.database import DatabaseManager
from ..config.config import OrigData, CSVOrigMeta
from ..config.config import ConfigReadError
from ..config.config import ConfigExprInterpreter
from ..config.definitions.origdata import META_FIELD_KEYS
from ..config.definitions.origdata import TAG_FLD_NM
from ..config.definitions.origdata import PARAM_FLD_NM
from ..logger import rawcsv
from ..environ import glob_var
from ..config.pattern import INSTR_TYPE_PAT
from ..config.definitions.tag import TAG_RAW_VAL, TAG_GDP_VAL, TAG_EMPTY_VAL


# Pandas csv_read method arguments
PD_CSV_READ_ARGS = [
    ('csv_' + arg) for arg in inspect.getfullargspec(pd.read_csv).args[1:]
]


class Handler(ABC):
    """
    The Handler interface declares a method for building the chain of handlers.
    It also declares a method for executing a request.

    Note:
        `Source <https://refactoring.guru/design-patterns/chain-of-responsibility/python/example>`__

    """

    @abstractmethod
    def set_next(self, handler):
        """Method to set next handler

        Args:
            handler (Handler): Handler class

        Returns:
            Handler
        """

    @abstractmethod
    def handle(self, request, prm_abbr):
        """Handle method

        Args:
            request (`object`): Request
            prm_abbr (str): Parameter abbr

        Returns:
            Optional: 'object'

        """


class AbstractHandler(Handler):
    """
    The default chaining behavior can be implemented inside a base handler
    class.

    .. uml::

        @startuml
        footer Chain of responsibility design pattern

        class AbstractHandler {
            set_next(handler)
            {abstract} handle
        }

        class FileHandler {
            {abstract} get_main(*args, **kwargs)
            {abstract} get_metadata_item(*args, **kwargs)
        }

        AbstractHandler <--o AbstractHandler
        AbstractHandler <|-- FileHandler : extends
        FileHandler <|-- CSVHandler
        FileHandler <|-- GDPHandler

        @enduml


    """

    def __init__(self):

        # Init attributes
        self._db_mngr = DatabaseManager()
        self._next_handler = None

    def set_next(self, handler):
        """
        Returning a handler from here will let us link handlers in a
        convenient way like this:
        handler1.set_next(handler2).set_next(handler3)

        """
        self._next_handler = handler

        return handler

    @abstractmethod
    def handle(self, *args):
        """Super handler behavior"""
        if self._next_handler:
            return self._next_handler.handle(*args)
        else:
            return


class FileHandler(AbstractHandler):
    """File handler"""

    def __init__(self):

        # Call super constructor
        super().__init__()

        # Set attributes
        self._origdata_config_mngr = OrigData()

        # Init origdata config manager
        self._origdata_config_mngr.read()

    @property
    def origdata_config_mngr(self):
        """config.config.OrigData: Yaml original metadata manager"""
        return self._origdata_config_mngr

    @property
    @abstractmethod
    def file_suffix(self):
        """re.compile : Handled file suffix (re.fullmatch of pathlib.Path.suffix)"""

    @property
    @abstractmethod
    def file_instr_type_pat(self):
        """re.compile : File instr_type pattern
        (re.search within pathlib.Path.name). Group #1 must correspond to
        instr_type name."""

    def handle(self, file_path, prm_abbr):
        """Handle method"""
        if self.file_suffix.fullmatch(file_path.suffix) is not None:
            return self.get_main(file_path, prm_abbr)
        else:
            return super().handle(file_path, prm_abbr)

    @abstractmethod
    def get_metadata_item(self, *args, **kwargs):
        """Method to get metadata item"""

    @abstractmethod
    def get_main(self, *args, **kwargs):
        """Main get method called from handle method"""

    def get_instr_type(self, file_path):
        """Get instrument type from file path

        Args:
            file_path:

        Returns:

        """

        if (grp := re.search(self.file_instr_type_pat, file_path.name)) is None:
            # TODO Detail exception
            raise Exception(
                f"Bad instr_type syntax in data file '{file_path}'"
            )

        # Get from group
        instr_type_name = grp.group(1)

        # Check instr_type name existence in DB
        if self._db_mngr.get_or_none(
                InstrType,
                search={
                    'where': InstrType.type_name == instr_type_name
                },
                attr=[[InstrType.type_name.name]]
        ) is None:
            # TODO Detail exception
            raise Exception(
                f"Missing instr_type '{instr_type_name}' in DB while reading " +
                f"data file '{file_path}'"
            )

        return instr_type_name

    # TODO
    #  Erase
    # def check_sn(self, file_path, instr_type_name, metadata):
    #     """Check serial number in DB"""
    #     # Check instr_type name existence
    #     # (need it for loading origdata config)
    #     if (
    #         instr_type_name_from_sn := self._db_mngr.get_or_none(
    #             Instrument,
    #             search={
    #                 'join_order': [InstrType],
    #                 'where': Instrument.srn == metadata[SRN_FLD_NM]
    #             },
    #             attr=[
    #                 [Instrument.instr_type.name, InstrType.type_name.name]
    #             ]
    #         )
    #     ) is None:
    #         # TODO Detail exception
    #         raise Exception(
    #             f"Missing instrument SN '{metadata[SRN_FLD_NM]}' in DB while reading " +
    #             f"data file '{file_path}'"
    #         )
    #
    #     if instr_type_name != instr_type_name_from_sn[0]:
    #         #TODO Detail exception
    #         raise Exception(
    #             f"Instrument SN '{metadata[SRN_FLD_NM]}' does not correspond to instr_type in " +
    #             f"('{instr_type_name}' != '{instr_type_name_from_sn}') " +
    #             f"for data file '{file_path}'"
    #         )

    def exclude_file(self, path_scan, prm_abbr):
        """Exclude file method"""

        # Search exclude file names source
        exclude_file_name = self._db_mngr.get_or_none(
            Info,
            search={
                'where': (
                    Parameter.prm_abbr == prm_abbr
                ),
                'join_order': [Parameter, DataSource]},
            attr=[[Info.data_src.name, DataSource.source.name]],
            get_first=False
        )

        origdata_path_new = [
            arg for arg in path_scan
            if self.get_source_unique_id(arg) not in exclude_file_name
        ]

        return origdata_path_new

    def read_metaconfig_fields(self, instr_type_name, prm_abbr):
        """Read field from metaconfig"""

        # Create metadata output
        out = {}
        for key in META_FIELD_KEYS:

            try:
                # TODO
                #  Consider if prm_abbr is mandatory at this point
                field_val = self.origdata_config_mngr.get_val(
                    [instr_type_name, prm_abbr], key
                )

                if isinstance(field_val, str):
                    meta_val = ConfigExprInterpreter.eval(
                        field_val, self.get_metadata_item
                    )

                else:
                    meta_val = [
                        ConfigExprInterpreter.eval(
                            field_val_arg, self.get_metadata_item
                        )
                        for field_val_arg in field_val
                    ]

                out.update({key: meta_val})

            #TODO Details exceptions
            except Exception as exc:
                raise Exception(f"{exc} / {key}")

        return out

    @staticmethod
    def get_source_unique_id(file_path):
        """Return string use to determine if a file have already be read.

        Args:
            file_path (pathlib.Path): Original file path

        Returns:
            int

        """

        # Get file name
        out = file_path.name

        return out


class CSVHandler(FileHandler):
    """CSV Hanlder class"""

    _FILE_SUFFIX = re.compile(r'\.csv', re.IGNORECASE)
    _FILE_INSTR_TYPE_PAT = re.compile(
        r"^(" + INSTR_TYPE_PAT + r")\.\w+"
    )

    def __init__(self):

        # Call super constructor
        super().__init__()

        # Define attributes
        self.cfg_file_suffix = [
            '.' + arg for arg in glob_var.config_file_ext
        ]
        self._origmeta_mngr = CSVOrigMeta()

    @property
    def origmeta_mngr(self):
        """config.config.CSVOrigMeta: Yaml original CSV file metadata manager"""
        return self._origmeta_mngr

    @property
    def file_suffix(self):
        return self._FILE_SUFFIX

    @property
    def file_instr_type_pat(self):
        return self._FILE_INSTR_TYPE_PAT

    def get_metadata_item(self, item):
        """Implementation of abstract method"""
        return self.origmeta_mngr[item]

    def get_metadata(self, file_path, instr_type_name, prm_abbr):
        """Method to get metadata"""

        # Init
        self.origmeta_mngr.init_document()

        # Define metadata file path
        # Check if data file with config suffix exist. If not, metadata
        # should be in data file
        try:
            metadata_file_path = next(
                arg for arg in file_path.parent.glob(
                    '*' + file_path.stem + '*.*'
                ) if arg.suffix in self.cfg_file_suffix
            )
        except StopIteration:
            metadata_file_path = file_path

        # Read metadata
        with metadata_file_path.open(mode='r') as fid:

            # Meta data are in data file
            if metadata_file_path == file_path:
                meta_raw = ''.join(
                    [arg[1:] for arg in
                     takewhile(lambda x: x[0] in ['#', '%'], fid)
                     ]
                )

            # Meta data are in separate file
            else:
                meta_raw = fid.read()

        # Read YAML config
        try:
            self.origmeta_mngr.read(meta_raw)
            assert self.origmeta_mngr.document is not None

        except ConfigReadError as exc:
            #TODO raise exception
            rawcsv.error(
                "Error in reading file '%s' (%s)",
                metadata_file_path,
                exc
            )
            return

        except AssertionError:
            # TODO raise exception
            rawcsv.error(
                "No meta data found in file '%s'",
                metadata_file_path
            )
            return

        # Read metadata fields
        try:
            out = self.read_metaconfig_fields(instr_type_name, prm_abbr)

        except Exception as exc:
            raise Exception(exc)

        return out

    def get_main(self, file_path, prm_abbr):
        """Implementation of abstract method"""

        # Get instr_type
        instr_type_name = self.get_instr_type(file_path)

        # Get metadata
        # (need instr_type_name
        if (
                metadata := self.get_metadata(
                    file_path, instr_type_name, prm_abbr
                )
        ) is None:
            return

        # TODO
        #  Erase
        # # Check instr_type name existence
        # # (need it for loading origdata config)
        # self.check_sn(file_path, instr_type_name, metadata)

        # Create info with 'raw' tag
        metadata[TAG_FLD_NM] += [TAG_RAW_VAL]

        # Get config params for (instr_type, prm_abbr) couple
        origdata_cfg_prm = self.origdata_config_mngr.get_all_default(
            [instr_type_name, prm_abbr]
        )

        # Get raw data config param
        raw_csv_read_args = {
            key.replace('csv_', ''): val for key, val in origdata_cfg_prm.items()
            if key in PD_CSV_READ_ARGS}

        # Add usecols
        try:
            # Set read_csv arguments
            raw_csv_read_args.update(
                {
                    'usecols': [
                        self.origdata_config_mngr.get_val(
                            [instr_type_name, prm_abbr], PARAM_FLD_NM
                        ),
                    ],
                    'squeeze': True,
                }
            )

            # Read raw csv
            data = pd.read_csv(file_path, **raw_csv_read_args)
            data = data.map(
                eval(self.origdata_config_mngr.get_val(
                    [instr_type_name, prm_abbr], 'lambda')
                )
            )

            # Log
            rawcsv.info(
                "Successful reading of '%s' in CSV file '%s'",
                prm_abbr, file_path,
            )

        except KeyError:

            # Create empty data set
            data = pd.Series([])

            # Add empty tag
            metadata[TAG_FLD_NM] += [TAG_EMPTY_VAL]

            # Log
            rawcsv.warn(
                "No data for '%s' in file '%s'",
                prm_abbr, file_path,
            )

        except ValueError as exc:
            raise OrigConfigError(
                f"Error while reading '{prm_abbr}' in file '{file_path}' " +
                f"({type(exc).__name__}: {exc})"
            )

        # Append data
        out = {
            'info': metadata,
            'prm_abbr': prm_abbr,
            'index': data.index.values,
            'value': data.values,
            'source_info': self.get_source_unique_id(file_path)
        }

        return out


class GDPHandler(FileHandler):
    """GDP Handler class"""

    _FILE_SUFFIX = re.compile(r'\.nc', re.IGNORECASE)
    _FILE_INSTR_TYPE_PAT = re.compile(
        r"^[A-Z]{3}\-[A-Z]{2}\-\d{2}\_\d\_([\w\-]+\_\d{3})\_\d{8}T"
    )

    def __init__(self):

        # Call super constructor
        super().__init__()

        # Set file id attribute
        self._fid = None

    @property
    def file_suffix(self):
        return self._FILE_SUFFIX

    @property
    def file_instr_type_pat(self):
        return self._FILE_INSTR_TYPE_PAT

    def get_metadata_item(self, item):
        """Implementation of abstract method"""
        return self._fid.getncattr(item)

    def get_metadata(self, file_path, instr_type_name, prm_abbr):
        """Method to get file metadata"""

        with nc.Dataset(file_path, 'r') as self._fid:

            # Read metadata fields
            try:
                out = self.read_metaconfig_fields(instr_type_name, prm_abbr)

            #TODO Detail exception
            except Exception as exc:
                raise Exception(f"{exc} / {instr_type_name} / {prm_abbr}")

        return out

    def get_main(self, file_path, prm_abbr):
        """Implementation of abstract method"""

        # Get instr_type
        instr_type_name = self.get_instr_type(file_path)

        # Get metadata
        metadata = self.get_metadata(
            file_path, instr_type_name, prm_abbr
        )

        # TODO
        #  Erase
        # # Check instr_type name existence
        # # (need it for loading origdata config)
        # self.check_sn(file_path, instr_type_name, metadata)

        # Create event with 'raw' and 'gdp' tag
        metadata[TAG_FLD_NM] += [TAG_RAW_VAL, TAG_GDP_VAL]

        try:

            # Read data
            with nc.Dataset(file_path, 'r') as self._fid:
                data_col_nm = self.origdata_config_mngr.get_val(
                    [instr_type_name, prm_abbr], PARAM_FLD_NM
                )

                data = pd.Series(self._fid[data_col_nm][:])
                data = data.map(
                    eval(self.origdata_config_mngr.get_val(
                        [instr_type_name, prm_abbr], 'lambda')
                    )
                )

        except KeyError:

            # Create empty data set
            data = pd.Series([])

            # Add empty tag
            metadata[TAG_FLD_NM] += [TAG_EMPTY_VAL]

            # Log
            rawcsv.warn(
                "No data for '%s' in file '%s'",
                prm_abbr, file_path,
            )

        #TODO Detail exception
        except Exception as exc:
            raise OrigConfigError(
                f"Error while reading '{prm_abbr}' in file '{file_path}' " +
                f"({type(exc).__name__}: {exc})"
            )

        # Append data
        out = {
            'info': metadata,
            'index': data.index.values,
            'value': data.values,
            'prm_abbr': prm_abbr,
            'source_info': self.get_source_unique_id(file_path)
        }

        return out


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

    def __init__(self):

        # Call super constructor
        super().__init__()

        # Init attributes
        self._db_mngr = DatabaseManager()

    def load(self, search, prm_abbr, filter_empty=True):
        """Load parameter method

        Args:
            search (str): Data loader search criterion
            prm_abbr (str): Positional parameter abbr
            filter_empty (bool, `optional`): Filter empty data from search.
                Default to True.

        Returns:
            list of dict

        .. uml::

            @startuml
            hide footbox

            LocalDBLinker -> DatabaseManager: get_data()
            LocalDBLinker <- DatabaseManager : data

            @enduml

        """

        # Retrieve data from DB
        data = self._db_mngr.get_data(
            search_expr=search, prm_abbr=prm_abbr, filter_empty=filter_empty
        )

        return data

    def save(self, data_list):
        """Save data method

        Args:
            data_list (list of dict): dict mandatory items are 'index' (np.array),
                'value' (np.array), 'info' (InfoManager|dic), 'prm_abbr' (str).
                dict optional key are 'source_info' (str), force_write (bool)

        """

        # Add data to DB
        for kwargs in data_list:
            self._db_mngr.add_data(**kwargs)


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


class ConfigInstrIdError(Exception):
    """Error for missing instrument id"""


class OutputDirError(Exception):
    """Error for bad output directory path"""


class OrigConfigError(Exception):
    """Error for bad orig config"""
