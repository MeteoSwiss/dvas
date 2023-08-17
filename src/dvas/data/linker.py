"""
Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data linker classes

"""

# Import external python packages and modules
import logging
from abc import ABC, abstractmethod
import re
from functools import reduce
from itertools import takewhile
import operator
import inspect
import netCDF4 as nc
import numpy as np
import pandas as pd

# Import from current package
from ..environ import path_var as env_path_var
from ..errors import DvasError
from ..database.model import Model as TableModel
from ..database.model import Info as TableInfo
from ..database.model import DataSource
from ..database.model import Prm as TableParameter
from ..database.database import DatabaseManager
from ..config.config import CSVOrigMeta
from ..config.config import ConfigGetError, ConfigReadError
from ..config.config import ConfigExprInterpreter
from ..config.definitions.origdata import EXPR_FIELD_KEYS
from ..config.definitions.origdata import TAG_FLD_NM
from ..config.definitions.origdata import VALUE_FLD_NM
from ..config.definitions.origdata import CSV_USE_DEFAULT_FLD_NM
from ..environ import glob_var
from ..hardcoded import GDP_FILE_EXT
from ..hardcoded import PRM_PAT, FLG_PRM_PAT
from ..hardcoded import CSV_FILE_MDL_PAT, GDP_FILE_MDL_PAT
from ..hardcoded import TAG_ORIGINAL, TAG_GDP, TAG_EMPTY
from ..tools import wmo

# Setup local logger
logger = logging.getLogger(__name__)

# Pandas csv_read method arguments
PD_CSV_READ_ARGS = [
    ('csv_' + arg) for arg in
    list(inspect.signature(pd.read_csv).parameters.keys())[1:]
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
    def handle(self, request, prm_name):
        """Handle method

        Args:
            request (`object`): Request
            prm_name (str): Parameter name

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

        return


class FileHandler(AbstractHandler):
    """File handler"""

    def __init__(self, orig_data_cfg):
        """
        Args:
            orig_data_cfg (config.config.OrigData): Original data config manager
        """

        # Call super constructor
        super().__init__()

        # Init attributes
        self._origdata_config_mngr = orig_data_cfg

        # Set default child defined attributes
        self._file_suffix_re = re.compile('')
        self._prm_re = re.compile('')
        self._file_model_pat = re.compile('')
        self._data_ok_tags = []

    @property
    def origdata_config_mngr(self):
        """config.config.OrigData: Yaml original metadata manager"""
        return self._origdata_config_mngr

    @property
    def file_suffix_re(self):
        """re.compile : Handled data file suffix."""
        return self._file_suffix_re

    @property
    def prm_re(self):
        """re.compile: Handled parameter name."""
        return self._prm_re

    @property
    def file_model_pat(self):
        """re.compile: File model pattern. Group #1 must correspond to model name."""
        return self._file_model_pat

    def check_file(self, file):
        """Check if file as the correct suffix pattern.

        Args:
            file (pathlib.Path): File path or file name

        Returns:
            bool: True if file name match

        """
        if self.file_suffix_re.fullmatch(file.suffix) is None:
            return False

        return True

    def check_prm(self, prm_name):
        """Check if parameter as correct parameter pattern

        Args:
            prm_name (str): Parameter name

        Returns:
            bool: True if file parameter name match

        """
        if self.prm_re.fullmatch(prm_name) is None:
            return False

        return True

    @property
    def data_ok_tags(self):
        """list of str: Tags list to add to metadata when data reading is successful"""
        return self._data_ok_tags

    def check_file_mdl(self, file):
        """Check if file name as correct model pattern

        Args:
            file (pathlib.Path): File path or file name

        Returns:
            bool: True if file parameter name match

        """
        if self.file_model_pat.match(file.name) is None:
            logger.debug('File %s does not match the pattern: %s',
                         file.name, self.file_model_pat)
            return False

        return True

    def handle(self, data_file_path, prm_name):
        """ Handle method

        Args:
            data_file_path (pathlib.Path): Data file path
            prm_name (str): Parameter name

        Returns:
            dict

        """

        if (
            self.check_file(data_file_path) and
            self.check_prm(prm_name) and
            self.check_file_mdl(data_file_path)
        ):
            return self._get_main(data_file_path, prm_name)

        else:
            return super().handle(data_file_path, prm_name)

    def _get_main(self, data_file_path, prm_name):
        """Main get method called from handle method"""

        # Get model
        mdl_name = self.get_model(data_file_path)

        # Get metadata
        # (need mdl_name to read config file)
        if (metadata := self.get_metadata(data_file_path, mdl_name, prm_name)) is None:
            # TODO
            #  Check this return
            return

        try:

            # Get parameter columns to read
            param_value = self.origdata_config_mngr.get_val(
                [mdl_name, prm_name], VALUE_FLD_NM
            )

            data = LoadExprInterpreter.eval(
                param_value, self.get_data, data_file_path, mdl_name, prm_name
            )

            # Add tags
            metadata[TAG_FLD_NM] += self.data_ok_tags

            # Log
            logger.debug("Successful reading of '%s' in file '%s'", prm_name, data_file_path)

        except ConfigGetError:

            # Create empty data set
            data = pd.Series([], dtype='float')

            # Add empty tag
            metadata[TAG_FLD_NM] += [TAG_EMPTY]

            # Log
            logger.debug("No data for '%s' in file '%s'", prm_name, data_file_path)

        except ValueError as exc:
            raise OrigConfigError(
                f"Error while reading '{prm_name}' in file '{data_file_path}' " +
                f"({type(exc).__name__}: {exc})"
            )

        # Add source
        # (use stem to have same hash for data file and flag file)
        metadata[DataSource.src.name] = self.get_source_unique_id(data_file_path)

        # Append data
        out = {
            'info': metadata,
            'prm_name': prm_name,
            'index': data.index.values,
            'value': data.values,
        }

        return out

    @abstractmethod
    def get_data(self, *args, **kwargs):
        """Method used to get data"""

    @abstractmethod
    def get_metadata_item(self, *args, **kwargs):
        """Method to get metadata item"""

    @abstractmethod
    def get_metadata_filename(self, data_file_path):
        """Method to get metadata file name"""

    @abstractmethod
    def get_metadata(self, file_path, mdl_name, prm_name):
        """Method to get metadata"""

        # Add automatic fields to metadata
        out = {TableModel.mdl_name.name: mdl_name}

        return out

    def get_model(self, file_path):
        """Get instrument type from file path

        Args:
            file_path:

        Returns:

        """

        # Init
        db_mngr = DatabaseManager()

        # Test
        if (grp := re.search(self.file_model_pat, file_path.name)) is None:
            raise DvasError(f"Bad model syntax in data file '{file_path}'")

        # Get from group
        mdl_name = grp.group(1)

        # Check model name existence in DB
        if db_mngr.get_or_none(
                TableModel,
                search={'where': TableModel.mdl_name == mdl_name},
                attr=[[TableModel.mdl_name.name]]
        ) is None:
            raise DvasError(f"Missing model '{mdl_name}' in DB while reading " +
                            f"data file '{file_path}'")

        return mdl_name

    def filter_files(self, path_list, prm_name):
        """Filter files already load.

        Args:
            path_list (pathlib.Path): List of file path to be load.
            prm_name (str): Corresponding parameter name.

        Returns:
            list

        """

        # Init
        db_mngr = DatabaseManager()

        # Search exclude file names source
        if (exclude_file_name := db_mngr.get_or_none(
            TableInfo,
            search={
                'where': (
                    TableParameter.prm_name == prm_name
                ),
                'join_order': [TableParameter, DataSource]},
            attr=[[TableInfo.data_src.name, DataSource.src.name]],
            get_first=False
        )) is None:
            out = path_list
        else:
            exclude_file_name = [arg[0] for arg in exclude_file_name]

            out = [
                arg for arg in path_list
                if not (
                    (self.get_source_unique_id(arg) in exclude_file_name) or
                    (arg.suffix in CSVHandler.CFG_FILE_SUFFIX)
                )
            ]

        return out

    def read_metaconfig_fields(self, mdl_name, prm_name):
        """Read field from metaconfig"""

        # Create metadata output
        out = {}
        for key in EXPR_FIELD_KEYS:

            try:
                # TODO
                #  Consider if prm_name is mandatory at this point
                field_val = self.origdata_config_mngr.get_val(
                    [mdl_name, prm_name], key
                )

                if isinstance(field_val, str):
                    meta_val = ConfigExprInterpreter.eval(
                        field_val, self.get_metadata_item
                    )

                elif isinstance(field_val, list):
                    meta_val = [
                        ConfigExprInterpreter.eval(
                            field_val_arg, self.get_metadata_item
                        )
                        for field_val_arg in field_val
                    ]

                elif isinstance(field_val, dict):

                    meta_val = {
                        field_val_key: ConfigExprInterpreter.eval(
                            field_val_val, self.get_metadata_item
                        )
                        for field_val_key, field_val_val in field_val.items()
                    }

                else:
                    raise OrigConfigError(
                        f"Field value '{field_val}' must be a of type (str, list, dict)")

                out.update({key: meta_val})

            except Exception as exc:
                raise DvasError(f"{exc} / {key}")

        return out

    @staticmethod
    def get_source_unique_id(file_path):
        """Return string use to determine if a file have already be read.

        Note:
            Stem is used to have same hash for data file and flag file)

        Args:
            file_path (pathlib.Path): Original file path

        Returns:
            int

        """

        # Get file name
        out = file_path.stem

        return out


class CSVHandler(FileHandler):
    """CSV Handler class"""

    CFG_FILE_SUFFIX = ['.' + arg for arg in glob_var.config_file_ext]

    def __init__(self, orig_data_cfg):
        """
        Args:
            orig_data_cfg (config.config.OrigData): Original data config manager
        """

        # Call super constructor
        super().__init__(orig_data_cfg)

        # Define attributes
        self._file_suffix_re = re.compile(
            rf'\.(({")|(".join(glob_var.csv_file_ext)}))',
            re.IGNORECASE
        )
        self._prm_re = re.compile(PRM_PAT)
        self._file_model_pat = re.compile(CSV_FILE_MDL_PAT)
        self._origmeta_mngr = CSVOrigMeta()

        self._data_ok_tags = [TAG_ORIGINAL]

    @property
    def origmeta_mngr(self):
        """config.config.CSVOrigMeta: Yaml original CSV file metadata manager"""
        return self._origmeta_mngr

    def get_metadata_item(self, item):
        """Implementation of abstract method"""
        return self.origmeta_mngr[item]

    def get_metadata_filename(self, data_file_path):
        """Implementation of abstract method"""
        # Check if data file with config suffix exist. If not, metadata
        # should be in data file
        try:
            metadata_file_path = next(
                arg for arg in data_file_path.parent.glob(
                    data_file_path.stem + '.*'
                ) if arg.suffix in self.CFG_FILE_SUFFIX
            )
        except StopIteration:
            metadata_file_path = data_file_path

        return metadata_file_path

    def get_metadata(self, file_path, mdl_name, prm_name):
        """Implementation of abstract method"""

        # Get default output from parent method
        out = super().get_metadata(file_path, mdl_name, prm_name)

        # Init
        self.origmeta_mngr.init_document()

        # Define metadata file path
        metadata_file_path = self.get_metadata_filename(file_path)

        # Read metadata
        with metadata_file_path.open(mode='r') as fid:

            # Meta data are in data file
            if metadata_file_path == file_path:
                meta_original = ''.join(
                    [arg[1:] for arg in
                     takewhile(lambda x: x[0] in ['#', '%'], fid)
                     ]
                )

            # Meta data are in separate file
            else:
                meta_original = fid.read()

        # Read YAML config
        try:
            self.origmeta_mngr.read(meta_original)
            assert self.origmeta_mngr.document is not None

        except ConfigReadError as exc:
            logger.error("Error in reading file '%s' (%s)", metadata_file_path, exc)
            return

        except AssertionError:
            logger.error("No meta data found in file '%s'", metadata_file_path)
            return

        # Read metadata fields
        try:
            out.update(self.read_metaconfig_fields(mdl_name, prm_name))

        except Exception as exc:
            raise Exception(exc)

        return out

    def get_data(self, field_id, data_file_path, mdl_name, prm_name):
        """Implementation of abstract method"""

        # Get config params for (model, prm_name) couple
        origdata_cfg_prm = self.origdata_config_mngr.get_all(
            [mdl_name, prm_name]
        )

        # Get original data config param
        original_csv_read_args = {key: val for key, val in origdata_cfg_prm.items()
                                  if key in PD_CSV_READ_ARGS}

        # Reset to default if dedicated config field is True
        if origdata_cfg_prm[CSV_USE_DEFAULT_FLD_NM]:
            original_csv_read_args_def = {
                key: val for key, val in self.origdata_config_mngr.get_default().items()
                if key in PD_CSV_READ_ARGS}
            original_csv_read_args.update(original_csv_read_args_def)

        # Replace prefix
        original_csv_read_args = {key.replace('csv_', ''): val
                                  for key, val in original_csv_read_args.items()}

        # Transform the skiprow argument into a suitable lambda function, if warranted.
        # That's to deal with side-effects of #182 when the list of column names is not the last
        # one before the data. fpavogt, 26.11.2021
        if 'skiprows' in original_csv_read_args.keys():
            if isinstance(original_csv_read_args['skiprows'], str):
                original_csv_read_args['skiprows'] = eval(original_csv_read_args['skiprows'])

        # Set read_csv arguments
        # (Add usecols, squeeze and engine arguments)
        original_csv_read_args.update({'usecols': [field_id], 'engine': 'python'})

        # Read original csv
        data = pd.read_csv(data_file_path, **original_csv_read_args).squeeze("columns")

        return data


class GDPHandler(FileHandler):
    """GDP Handler class"""

    def __init__(self, orig_data_cfg):
        """
        Args:
            orig_data_cfg (config.config.OrigData): Original data config manager
        """

        # Call super constructor
        super().__init__(orig_data_cfg)

        # Set file id attribute
        self._file_suffix_re = re.compile(rf'\.{GDP_FILE_EXT}', re.IGNORECASE)
        self._prm_re = re.compile(PRM_PAT)
        self._file_model_pat = re.compile(GDP_FILE_MDL_PAT)
        self._fid = None

        self._data_ok_tags = [TAG_ORIGINAL, TAG_GDP]

    def get_metadata_item(self, item):
        """Implementation of abstract method"""
        return self._fid.getncattr(item)

    def get_metadata_filename(self, data_file_path):
        """Implementation of abstract method"""
        return data_file_path

    def get_metadata(self, file_path, mdl_name, prm_name):
        """Method to get file metadata"""

        # Get default output from parent method
        out = super().get_metadata(file_path, mdl_name, prm_name)

        # Define metadata file path
        metadata_file_path = self.get_metadata_filename(file_path)

        with nc.Dataset(metadata_file_path, 'r') as self._fid:  #noqa pylint: disable=no-member

            # Read metadata fields
            try:
                out.update(
                    self.read_metaconfig_fields(mdl_name, prm_name)
                )

            # TODO Detail exception
            except Exception as exc:
                raise DvasError(f"{exc} / {mdl_name} / {prm_name}")

        return out

    def get_data(self, field_id, data_file_path, mdl_name, prm_name):
        """Implementation of abstract method"""

        # Read data
        with nc.Dataset(data_file_path, 'r') as self._fid:  #noqa pylint: disable=no-member
            data = pd.Series(self._fid[field_id][:])

        return data


class FlgCSVHandler(CSVHandler):
    """CSV flag file handler class"""

    def __init__(self, orig_data_cfg):
        """
        Args:
            orig_data_cfg (config.config.OrigData): Original data config manager
        """

        # Call super constructor
        super().__init__(orig_data_cfg)

        # Define attributes
        self._file_suffix_re = re.compile(
            rf'\.(({")|(".join(glob_var.flg_file_ext)}))',
            re.IGNORECASE
        )
        self._prm_re = re.compile(FLG_PRM_PAT)


class FlgGDPHandler(GDPHandler):
    """GDP flag file handler class"""

    def __init__(self, orig_data_cfg):
        """
        Args:
            orig_data_cfg (config.config.OrigData): Original data config manager
        """

        # Call super constructor
        super().__init__(orig_data_cfg)

        # Define attributes
        self._file_suffix_re = re.compile(
            rf'\.(({")|(".join(glob_var.flg_file_ext)}))',
            re.IGNORECASE
        )
        self._prm_re = re.compile(FLG_PRM_PAT)

        self._data_ok_tags = [TAG_ORIGINAL, TAG_GDP]

    def get_metadata_filename(self, data_file_path):
        """Implementation of abstract method"""
        return data_file_path.parent / (data_file_path.stem + f'.{GDP_FILE_EXT}')

    def get_data(self, field_id, data_file_path, mdl_name, prm_name):
        """Implementation of abstract method"""

        # Get config params for (model, prm_name) couple
        origdata_cfg_prm = self.origdata_config_mngr.get_all(
            [mdl_name, prm_name]
        )

        # Get original data config param
        original_csv_read_args = {key.replace('csv_', ''): val
                                  for key, val in origdata_cfg_prm.items()
                                  if key in PD_CSV_READ_ARGS}

        # Set read_csv arguments
        # (Add usecols, squeeze and engine arguments)
        original_csv_read_args.update({'usecols': [field_id], 'engine': 'python'})

        # Read original csv
        data = pd.read_csv(data_file_path, **original_csv_read_args).squeeze("columns")

        return data


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

    def load(self, search, prm_name, filter_empty=True):
        """Load parameter method

        Args:
            search (str): Data loader search criterion
            prm_name (str): Parameter name
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

        # Init
        db_mngr = DatabaseManager()

        # Retrieve data from DB
        data = db_mngr.get_data(
            search_expr=search, prm_name=prm_name, filter_empty=filter_empty
        )

        return data

    def save(self, data_list):
        """Save data method

        Args:
            data_list (list of dict): dict mandatory items are 'index' (np.array),
                'value' (np.array), 'info' (InfoManager|dict), 'prm_name' (str).
                dict optional key are 'source_info' (str), force_write (bool)

        """

        # Init
        db_mngr = DatabaseManager()

        # Add data to DB
        for kwargs in data_list:
            db_mngr.add_data(**kwargs)


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

        # Test
        if env_path_var.output_path is None:
            # TODO
            #  Detail exception
            raise Exception()

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


class LoadExprInterpreter(ABC):
    """Abstract config expression interpreter class

        Notes:
            This class and subclasses construction are based on the interpreter
            design pattern.

        """

    _FCT = None
    _ARGS = tuple()
    _KWARGS = dict()

    @classmethod
    def set_callable(cls, fct, *args, **kwargs):
        """Set strategy
        Args:
            fct (callable): Function/Methode called by 'get' expression
        """

        # Test
        assert callable(fct), "'fct' must be a callable"

        cls._FCT = fct
        cls._ARGS = args
        cls._KWARGS = kwargs

    @abstractmethod
    def interpret(self):
        """Interpreter method"""

    @staticmethod
    def eval(expr, get_fct, *args, **kwargs):
        r""" Evaluate str expression

        Args:
            expr (str|ConfigExprInterpreter): Expression to evaluate
            get_fct (callable): Function use by 'get'

        Examples:
            >>> import re
            >>> mymatch = re.match('^a(\d)', 'a1b')
            >>> print(ConfigExprInterpreter.eval("cat('My test', ' ', get(1))", mymatch.group))
            My test 1
        """

        # Define
        str_expr_dict = {
            'add': AddExpr,
            'sub': SubExpr,
            'mul': MulExpr,
            'div': DivExpr,
            'get': GetExpr,
            'pow': PowExpr,
            'sqrt': SqrtExpr,
            'getreldt': GetreldtExpr,
            'getgeomalt': GetgeomaltExpr,
        }

        # Init
        LoadExprInterpreter.set_callable(get_fct, *args, **kwargs)

        # Treat expression
        try:
            # Eval
            expr_out = eval(expr, str_expr_dict)

            # Interpret
            expr_out = expr_out.interpret()

        # TODO
        #  Detail exception
        except Exception as exp:
            raise DvasError(exp)

        return expr_out


class NonTerminalLoadExprInterpreter(LoadExprInterpreter):
    """Implement an interpreter operation for non terminal symbols in the
    grammar.
    """

    def __init__(self, *args):
        self._expression = args

    def interpret(self):
        """Non terminal interpreter method"""

        # Apply interpreter
        res_interp = [
            (arg if isinstance(arg, LoadExprInterpreter)
             else NoneExpr(arg)).interpret()
            for arg in self._expression
        ]

        if len(self._expression) > 1:
            return reduce(self.fct, res_interp)

        return self.fct(res_interp[0])

    @abstractmethod
    def fct(self, *args):
        """Function between expression args"""


class AddExpr(NonTerminalLoadExprInterpreter):
    """Addition"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.add(a, b)


class SubExpr(NonTerminalLoadExprInterpreter):
    """Subtractions"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.sub(a, b)


class MulExpr(NonTerminalLoadExprInterpreter):
    """Multiplication"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.mul(a, b)


class DivExpr(NonTerminalLoadExprInterpreter):
    """Division"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.truediv(a, b)


class PowExpr(NonTerminalLoadExprInterpreter):
    """Power operator"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.pow(a, b)


class SqrtExpr(PowExpr):
    """Square root"""

    def __init__(self, arg):
        super().__init__(arg, 0.5)


class TerminalLoadExprInterpreter(LoadExprInterpreter):
    """Implement an interpreter operation for terminal symbols in the
    grammar.
    """

    def __init__(self, arg):
        self._expression = arg


class NoneExpr(TerminalLoadExprInterpreter):
    """Apply none interpreter"""

    def interpret(self):
        """Implement fct method"""
        return self._expression


class GetExpr(TerminalLoadExprInterpreter):
    """Get catch value"""

    _CONV_DICT = {
        'nop': lambda x: x,
        'rel': lambda x: x - x.iloc[0],
        'div2': lambda x: x / 2,
        'c2k': lambda x: x + 273.15,
        'k2c': lambda x: x - 273.15,
        'c2f': lambda x: (x * 9 / 5) + 32,
        'f2c': lambda x: (x - 32) * 5 / 9,
        'ms2kmh': lambda x: x * 3.6,
        'kmh2ms': lambda x: x / 3.6,
        'm2km': lambda x: x / 1000,
        'km2m': lambda x: x + 1000,
        'kn2ms': lambda x: x * 1852 / 3600,
    }

    def __init__(self, arg, op='nop'):
        self._expression = arg

        # Set op
        if isinstance(op, str):
            self._op = [self._CONV_DICT[op]]
        else:
            self._op = [self._CONV_DICT[arg] for arg in op]

    def interpret(self):
        """Implement fct method"""
        out = self._FCT(self._expression, *self._ARGS, **self._KWARGS)  # noqa pylint: disable=E1102

        for op in self._op:
            out = op(out)

        return out


class GetgeomaltExpr(TerminalLoadExprInterpreter):
    """ Geometric altitude to geopotential height convertor """

    def __init__(self, arg, lat=None):
        """ Init function

        Args:
            args (str): expression to process.
            lat (float, optional): geodetic latitude of launch site, in degrees

        """

        self._expression = arg

        if lat is None:
            raise DvasError('Missing latitude for geopotential height conversion.')
        self._lat = np.deg2rad(lat)

    def interpret(self):
        """ Implement fct method """

        out = self._FCT(self._expression, *self._ARGS, **self._KWARGS)  # noqa pylint: disable=E1102

        # Convert geometric altitude to geopotential height
        # See #242 for details

        return wmo.geom2geopot(out, self._lat)


class GetreldtExpr(TerminalLoadExprInterpreter):
    """ Absolute datetimes to relative seconds """

    def __init__(self, arg, fmt=None, round_lvl=None):
        """ Initialization function

        Args:
            args (str): expression to process.
            fmt (str, optional): specify the datetime str format. Defaults to None.
            round_lvl (int, optional): Specify the time step rounding level,
                as (1/10)**round_lvl seconds. Defaults to None = full accuracy.

        Note:
            If set, the round_lvl parameter will be fed to the `decimals` argument of the
            `pandas.round()` routine. If the rounding leads to an error larger than
            (1/10)**(round_lvl+1) seconds, a critical log message will be created.

        """
        self._expression = arg

        if fmt is None:
            raise DvasError(f'Missing datetime decoding format in {self.__class__.__name__}')
        if not isinstance(fmt, str):
            raise DvasError(f'Datetime decoding format should be of type str, not: {type(fmt)}')
        self._fmt = fmt

        if not (round_lvl is None or isinstance(round_lvl, int)):
            raise DvasError(f'round_lvl should be of type "int", not: {type(round_lvl)}')
        self._round_lvl = round_lvl

    def interpret(self):
        """ Implement fct method """

        out = self._FCT(self._expression, *self._ARGS, **self._KWARGS)  # noqa pylint: disable=E1102

        # Convert the datetime, and get the time steps in s
        out = pd.to_datetime(out, format=self._fmt)
        out = (out - out.iloc[0]).apply(lambda x: x.total_seconds())
        # WARNING: in the line above, we do not use .dt.total_seconds(), because this can lead to
        # floating point errors ! See https://github.com/pandas-dev/pandas/issues/34290

        # If warranted, round the time steps
        if self._round_lvl is not None:
            out_orig = out.copy()
            out = out.round(decimals=self._round_lvl)

            # Raise a critical log message if rounding leads to errors larger than 1/10s of the
            # rounding level.
            if ((errs := (out-out_orig).abs()) >= 1/10**(self._round_lvl+1)).any():
                msg_lvl = logger.warning
            else:
                msg_lvl = logger.info

            msg_lvl('Maximum time stamp rounding error: %.3fs', errs.max())
            msg_lvl('Median (original) time step: %.3fs', np.median(np.diff(out_orig)))

        return out


class ConfigInstrIdError(Exception):
    """Error for missing instrument id"""


class OutputDirError(Exception):
    """Error for bad output directory path"""


class OrigConfigError(Exception):
    """Error for bad orig config"""
