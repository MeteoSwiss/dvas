"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Local database management tools

"""

# Import from python packages
import pprint
import re
from abc import abstractmethod, ABCMeta
from itertools import chain, zip_longest
import operator
from functools import wraps, reduce
from math import floor
from threading import Thread
from datetime import datetime
from peewee import chunked, DoesNotExist
from peewee import IntegrityError
from peewee import PeeweeException
from playhouse.shortcuts import model_to_dict
import numpy as np
from pandas import Timestamp
from pampy.helpers import Iterable, Union
import sre_yield

# Import from current package
from .model import db
from .model import Instrument, InstrType, Info
from .model import Parameter, Flag, DataSource, Data
from .model import Tag, InfosTags, InfosInstruments
from ..config.config import instantiate_config_managers
from ..config.config import InstrType as CfgInstrType
from ..config.config import Instrument as CfgInstrument
from ..config.config import Parameter as CfgParameter
from ..config.config import Flag as CfgFlag
from ..config.config import Tag as CfgTag
from ..config.definitions.tag import TAG_EMPTY_VAL
from ..dvas_helper import ContextDecorator
from ..dvas_helper import SingleInstanceMetaClass
from ..dvas_helper import TypedProperty as TProp
from ..dvas_helper import TimeIt
from ..dvas_helper import get_by_path, check_datetime
from ..dvas_helper import unzip, get_dict_len
from ..dvas_logger import localdb
from ..dvas_environ import glob_var
from ..dvas_environ import path_var as env_path_var


# Define
SQLITE_MAX_VARIABLE_NUMBER = 999
INDEX_NM = Data.index.name
VALUE_NM = Data.value.name

#: int: Database cache size in kB
DB_CACHE_SIZE = 10 * 1024

#: str: Local database file name
DB_FILE_NM = 'local_db.sqlite'


class OneDimArrayConfigLinker:
    """Link to OneDimArrayConfigManager
    config managers."""

    CFG_MNGRS = [CfgParameter, CfgInstrType, CfgInstrument, CfgFlag, CfgTag]

    def __init__(self, cfg_mngrs=None):
        """
        Args:
            cfg_mngrs (list of OneDimArrayConfigManager): Config managers
        """

        if cfg_mngrs is None:
            cfg_mngrs = self.CFG_MNGRS

        # Set attributes
        self._cfg_mngr = instantiate_config_managers(*cfg_mngrs)

    def get_document(self, key):
        """Interpret the generator syntax if necessary and
        return the config document.

        The generator syntax is apply only in OneDimArrayConfig  with
        not empty NODE_GEN. Generator sytax is based on regexpr. Fields which
        are not generator can contain expression to be evaluated. Expressions
        must be surrouded by '$'. To catch regexpr group in expression use
        'lambda x: x.group(N)'.

        Args:
            key (str): Config manager key

        Returns:
            dict

        Raises:
            - ConfigGenMaxLenError: Error for to much generated items.

        """

        def get_grp_fct(grp_fct):
            """Get group function as callable or str"""
            try:
                out = eval(grp_fct, {})
            except (NameError, SyntaxError):
                out = grp_fct
            return out

        # Init
        sep = glob_var.config_gen_grp_sep
        pat_spilt = r'\{0}[^\n\r\t\{0}]+\{0}'.format(sep)
        pat_find = r'\{0}([^\n\r\t{0}]+)\{0}'.format(sep)

        # Define
        array_old = self._cfg_mngr[key].document
        node_gen = self._cfg_mngr[key].NODE_GEN
        array_new = []

        # Loop over te config array items
        for doc in array_old:

            # Test if node generator allowed
            if node_gen:

                # Init new sub dict
                sub_dict_new = {}

                # Generate from regexp generator
                node_gen_val = sre_yield.AllMatches(doc[node_gen])

                # Check length
                if (n_val := len(node_gen_val)) > glob_var.config_gen_max:
                    raise ConfigGenMaxLenError(
                        f"{n_val} generated config field. " +
                        f"Max allowed {glob_var.config_gen_max}"
                    )

                # Update sub dict
                sub_dict_new.update({node_gen: list(node_gen_val)})

                # Loop over other config item key
                for key in filter(lambda x: x != node_gen, doc.keys()):

                    # Update new sub dict for current key
                    sub_dict_new.update(
                        {
                            key: [
                                ''.join(
                                    [arg for arg in chain(
                                        *zip_longest(

                                            # Split formula
                                            re.split(pat_spilt, doc[key]),

                                            # Find formula and substitute
                                            [
                                                re.sub(
                                                    doc[node_gen],
                                                    get_grp_fct(grp_fct),
                                                    node_gen_val[i].group()
                                                ) for grp_fct in
                                                re.findall(pat_find, doc[key])
                                            ]
                                        )
                                    ) if arg]
                                )
                                # Test if groups exists in generated str
                                if node_gen_val[i].groups() else
                                doc[key]
                                for i in range(len(node_gen_val))
                            ]
                        }
                    )

                # Rearange dict of list in list of dict
                res = [
                    dict(zip(sub_dict_new, arg))
                    for arg in zip(*sub_dict_new.values())
                ]

            # Case without generator
            else:
                res = [doc]

            # Append to new array
            array_new += res

        return array_new


class DatabaseManager(metaclass=SingleInstanceMetaClass):
    """Local data base manager.

    Note:
        If the data base does not exists, the creation will be forced.

    """

    DB_TABLES = [
        Info,
        InfosInstruments, Instrument, InstrType,
        InfosTags, Tag,
        DataSource,
        Data,
        Parameter,
        Flag,
    ]
    DB_TABLES_PRINT = [
        Parameter, InstrType,
        Instrument, Flag,
        Tag
    ]

    def __init__(self, reset_db=False):
        """
        Args:
            reset_db (bool, optional): Force the data base to be reset.
                Defaults to False.

        """

        # Create config linker instance attribute
        self._cfg_linker = OneDimArrayConfigLinker()

        # Create db attribute
        self._db = db

        # Init db
        db_new = self._init_db()

        # Create table
        self._create_tables()

        if reset_db or db_new:
            self._delete_tables()
            self._fill_metadata()

    @property
    def db(self):
        """peewee.SqliteDatabase: Database instance"""
        return self._db

    def _init_db(self):
        """Init db. Create new file if missing or take existing file.

        Returns:
            bool: True if the DB is newly created

        """

        # Define
        file_path = env_path_var.local_db_path / DB_FILE_NM
        pragmas = {
            'foreign_keys': True,
            # Set cache to 10MB
            'cache_size': -DB_CACHE_SIZE
        }

        # Create local DB directory
        if file_path.exists():
            db_new = False

        else:
            db_new = True
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                # Set user read/write permission
                file_path.parent.chmod(
                    file_path.parent.stat().st_mode | 0o600
                )
            except (OSError,) as exc:
                raise DBDirError(
                    f"Error in creating '{self._db.database.parent}' ({exc})"
                )

        # Init DB
        self._db.init(file_path, pragmas=pragmas)

        return db_new

    def get_or_none(self, table, search=None, attr=None, get_first=True):
        """Get from DB

        Args:
            table (peewee.Model):
            search (dict): Keys ['join_order', 'where']
            attr (list of (list of str)): Get query result attributes by path
            get_first (bool): return first occurrence or all. Default ot False.

        """

        # Init
        if search is None:
            search = {}

        if attr is None:
            attr = []

        qry = table.select()
        if search:

            if 'join_order' not in search.keys():
                search['join_order'] = []

            for jointbl in search['join_order']:
                qry = qry.join(jointbl)
                qry = qry.switch(table)
            qry = qry.where(search['where'])

        try:
            with DBAccess(self) as _:
                if get_first:
                    if attr:
                        out = [get_by_path(qry.get(), arg) for arg in attr]
                    else:
                        out = qry.get()

                else:
                    if attr:
                        out = [
                            [get_by_path(qry_arg, arg) for arg in attr]
                            for qry_arg in qry.iterator()
                        ]
                    else:
                        out = list(qry.iterator())

        except DoesNotExist:
            out = None

        return out

    @staticmethod
    def _model_to_dict(query, recurse=False):
        """Convert a query to a dictionary

        Notes:
            Must be used in a DB context manager

        Args:


        Returns:
            dict

        """
        return [model_to_dict(qry, recurse=recurse) for qry in query]

    def _create_tables(self):
        """Create table (safe mode)"""
        with DBAccess(self) as _:
            for table in self.DB_TABLES:
                table.create_table(safe=True)

    def _delete_tables(self):
        """Delete table instances"""
        with DBAccess(self) as _:
            for table in self.DB_TABLES:
                qry = table.delete()
                qry.execute()  # noqa pylint: disable=E1120

    def _fill_metadata(self):
        """Create db tables"""

        with DBAccess(self) as _:

            try:

                # Fill simple tables
                for tbl in [Parameter, InstrType, Flag, Tag]:
                    self._fill_table(tbl)

                # File instruments
                self._fill_table(
                    Instrument,
                    foreign_constraint=[
                        {
                            'attr': Instrument.instr_type.name,
                            'class': InstrType,
                            'foreign_attr': InstrType.type_name.name
                        },
                    ]
                )

            except IntegrityError as exc:
                raise DBCreateError(exc)

    def _fill_table(self, table, foreign_constraint=None):
        """

        Args:
            table:
            foreign_constraint:

        """

        # Init
        document = self._cfg_linker.get_document(table.__name__)

        # get foreign constraint attribute
        if foreign_constraint:
            for doc in document:
                for arg in foreign_constraint:
                    mdl_cls = arg['class']
                    cmp_res = getattr(mdl_cls, arg['foreign_attr']) == doc[arg['attr']]
                    doc[arg['attr']] = mdl_cls.get_or_none(cmp_res)

        # Test if document is empty
        if document:

            # Calculate max batch size
            n_max = floor(SQLITE_MAX_VARIABLE_NUMBER / get_dict_len(document[0]))

            # Insert to db
            for batch in chunked(document, n_max):
                table.insert_many(batch).execute()

        else:
            pass

            # TODO
            #  Log

    def _get_table(self, table, search=None, recurse=False):
        """

        Args:
            table:
            search: [join_order (optional), where]

        Returns:
            dict:

        """

        with DBAccess(self) as _:
            qry = table.select()
            if search:

                if 'join_order' not in search.keys():
                    search['join_order'] = []

                for jointbl in search['join_order']:
                    qry = qry.join(jointbl)
                    qry = qry.switch(table)
                qry = qry.where(search['where'])
            out = self._model_to_dict(qry, recurse=recurse)

        return out

    def add_data(
            self, index, value, info, prm_abbr,
            source_info=None, force_write=False
    ):
        """Add profile data to the DB.

        Args:
            index (np.array of int): Data index
            value (np.array of float): Data value
            info (InfoManager): Data info
            prm_abbr (str):
            source_info (str, optional): Data source
            force_write (bool, optional): force rewrite of already save data

        Raises:
            DBInsertError: Error while add data

        """

        # Test input
        assert len(index) == len(value),\
            "Data index and data value are of different length"
        assert isinstance(index, np.ndarray), "Data index is not an np.ndarray"
        assert isinstance(value, np.ndarray), "Data value is not an np.ndarray"

        # Add data
        try:
            with DBAccess(self) as _:

                # Get/Check instrument
                try:
                    instr_id_list = [
                        arg[0] for arg in
                        self.get_or_none(
                            Instrument,
                            search={
                                'where': Instrument.srn.in_(info.srn)
                            },
                            attr=[[Instrument.id.name]],
                            get_first=False
                        )
                    ]
                    assert len(instr_id_list) == len(info.srn)

                except AssertionError:
                    raise DBInsertError(
                        f"Many instrument srn in {info.srn} are missing in DB",
                    )

                # Get/Check parameter
                param = Parameter.get_or_none(
                    Parameter.prm_abbr == prm_abbr
                )
                if not param:
                    err_msg = "prm_abbr '%s' is missing in DB" % (prm_abbr)
                    localdb.error(err_msg)
                    raise DBInsertError(err_msg)

                # Check tag_txt existence
                try:
                    tag_id_list = [
                        arg[0] for arg in
                        self.get_or_none(
                            Tag,
                            search={
                                'where': Tag.tag_txt.in_(info.tags)
                            },
                            attr=[[Tag.id.name]],
                            get_first=False
                        )
                    ]
                    assert len(tag_id_list) == len(info.tags)

                except AssertionError:
                    raise DBInsertError(
                        f"Many tags in {info.tags} are missing in DB",
                    )

                # Create original data information
                data_src, _ = DataSource.get_or_create(
                    source=source_info, source_hash=hash(source_info))

                # Create info
                info_id, created = Info.get_or_create(
                    evt_dt=info.evt_dt, param=param,
                    data_src=data_src, evt_hash=hash(info)
                )

                # Erase data (created == False indicate that data already exists)
                if (created is False) and (force_write is True):

                    # Delete InfosTags entries
                    InfosTags.delete().\
                        where(InfosTags.info == info_id).\
                        execute()

                    # Delete InfosInstruments entries
                    InfosInstruments.delete().\
                        where(InfosInstruments.info == info_id).\
                        execute()

                    # Delete Data entries
                    Data.delete().\
                        where(Data.info == info)

                    # TODO
                    #  Add log

                # Insert data (created == True indicate that data are new)
                if (created is True) or (force_write is True):

                    # Link info to tag
                    tag_info = [
                        {
                            InfosTags.tag.name: tag_id,
                            InfosTags.info.name: info_id
                        } for tag_id in tag_id_list
                    ]
                    if tag_info:

                        # Calculate max batch size
                        n_max = floor(SQLITE_MAX_VARIABLE_NUMBER/get_dict_len(tag_info[0]))

                        # Insert
                        for batch in chunked(tag_info, n_max):
                            InfosTags.insert_many(batch).execute()  # noqa pylint: disable=E1120

                    # Link info to instrument
                    instr_info = [
                        {
                            InfosInstruments.instr.name: instr_id,
                            InfosInstruments.info.name: info_id
                        } for instr_id in instr_id_list
                    ]
                    if instr_info:

                        # Calculate max batch size
                        n_max = floor(SQLITE_MAX_VARIABLE_NUMBER / get_dict_len(instr_info[0]))

                        # Insert
                        for batch in chunked(instr_info, n_max):
                            InfosInstruments.insert_many(batch).execute()  # noqa pylint: disable=E1120

                    # Create batch index
                    fields = [Data.index, Data.value, Data.info]

                    # Create batch data
                    batch_data = zip(
                        index,
                        value,
                        [info_id] * len(value)
                    )

                    # Calculate max batch size
                    n_max = floor(SQLITE_MAX_VARIABLE_NUMBER / len(fields))

                    # Insert to db
                    for batch in chunked(batch_data, n_max):
                        Data.insert_many(batch, fields=fields).execute()  # noqa pylint: disable=E1120

                    # TODO
                    #  Add log

        except DBInsertError as exc:
            raise DBInsertError(exc)

    @staticmethod
    def _get_info_id(where_arg, prm_abbr, filter_empty):
        """Get info id"""

        try:
            out = list(SearchInfoExpr.eval(where_arg, prm_abbr, filter_empty))

        # TODO Detail exception
        except Exception as exc:
            print(exc)
            # TODO Decide if raise or not
            out = []

        return out

    @TimeIt()
    def get_data(self, where, prm_abbr, filter_empty):
        """Get data from DB

        Args:
            where:
            prm_abbr (str): Parameter
            filter_empty (bool): Filter empty data or not

        Returns:

        """

        # Get info id
        info_id_list = self._get_info_id(where, prm_abbr, filter_empty)

        if not info_id_list:
            localdb.warning(
                "Empty search '%s' for '%s", where, prm_abbr
            )

        # Query data
        qryer = []

        for info_id in info_id_list:
            qryer.append(Queryer(self, info_id))

        with DBAccess(self) as _:
            for qry in qryer:
                qry.start()
                qry.join()

            # Group data
            out = []
            for i, qry in enumerate(qryer):
                tag_txt_list = [
                    arg.tag_txt for arg in
                    Tag.select().distinct().
                    join(InfosTags).join(Info).
                    where(Info.id == info_id_list[i].id).
                    iterator()
                ]
                srn_list = [
                    arg.srn for arg in
                    Instrument.select().distinct().
                    join(InfosInstruments).join(Info).
                    where(Info.id == info_id_list[i].id).
                    iterator()
                ]
                out.append(
                    {
                        'info': InfoManager(
                            evt_dt=info_id_list[i].evt_dt,
                            srn=srn_list,
                            tags=tag_txt_list,
                        ),
                        'index': qry.index,
                        'value': qry.value,
                    }
                )

        return out

    def __str__(self, recurse=False, print_tables=None):
        """

        Args:
            recurse:
            print_tables:

        Returns:

        """

        # Init
        out = "Database content\n"
        out += f"{'*' * len(out)}\n"

        if not print_tables:
            print_tables = self.DB_TABLES_PRINT

        for print_tbl in print_tables:
            out += f"{print_tbl.__name__}\n"
            for arg in self._get_table(print_tbl, recurse=recurse):
                out += f"{arg}\n"
            out += "\n"

        return out

    def get_flags(self):
        """Get config flags

        Returns:
            list

        """
        return self._get_table(Flag)


class DBAccess(ContextDecorator):
    """Local SQLite data base context decorator"""

    def __init__(self, db_mngr, close_by_exit=True):
        """Constructor

        Args:
            db_mngr (DatabaseManager): DB manager instance
            close_by_exit (bool): Close DB by exiting context manager.
                Default to True

        """
        super().__init__()
        self._close_by_exit = close_by_exit
        self._transaction = None
        self._db_mngr = db_mngr

    def __call__(self, func):
        """Overwrite class __call__ method

        Args:
            func (callable): Decorated function

        """
        @wraps(func)
        def decorated(*args, **kwargs):
            with self as transaction:
                try:
                    out = func(*args, **kwargs)
                except PeeweeException:
                    transaction.rollback()
                    out = None
                return out
        return decorated

    def __enter__(self):
        """Class __enter__ method"""
        self._db_mngr.db.connect(reuse_if_open=True)
        self._transaction = self._db_mngr.db.atomic()
        return self._transaction

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Class __exit__ method"""
        if self._close_by_exit:
            self._db_mngr.db.close()


#TODO
# Test this db access context manager to possible speed up data select/insert
class DBAccessQ(ContextDecorator):
    """Data base context decorator"""

    def __init__(self, db):
        """Constructor

        Args:
            db (peewee.SqliteDatabase): PeeWee Sqlite DB object

        """
        super().__init__()
        self._db = db

    def __call__(self, func):
        """Overwrite class __call__ method"""
        @wraps(func)
        def decorated(*args, **kwargs):
            with self:
                try:
                    return func(*args, **kwargs)
                except PeeweeException as exc:
                    print(exc)

        return decorated

    def __enter__(self):
        """Class __enter__ method"""
        self._db.start()
        self._db.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Class __exit__ method"""
        self._db.close()
        self._db.stop()


class Queryer(Thread):
    """Data queryer

    Attributes:
        index (list): Data index
        value (list): Data value
        exc (Exception): Exception if occurs

    """

    def __init__(self, db_mngr, info_id):

        # Call super constructor
        super().__init__()

        # Init attributes
        self._info_id = info_id
        self._db_mngr = db_mngr
        self.index = None
        self.value = None
        self.exc = None

    def run(self):
        """Run method"""

        try:
            qry = (
                Data.
                select(Data.index, Data.value).
                where(Data.info == self._info_id)
            )

            with DBAccess(self._db_mngr, close_by_exit=False):

                data = list(unzip(qry.tuples().iterator()))

            # Set result
            self.index = data[0]
            self.value = data[1]

        except Exception as ex:
            self.exc = Exception(ex)

    def join(self, timeout=None):
        """Overwrite join method"""
        super().join(timeout=None)

        # Test exception attribute
        if self.exc:
            raise self.exc


class InfoManager:
    """Data info manager"""

    #: datetime.datetime: UTC datetime
    evt_dt = TProp(Union[str, Timestamp, datetime], check_datetime)

    #: str: Instrument id
    srn = TProp(
        Union[str, Iterable[str]],
        setter_fct=lambda x: (x,) if isinstance(x, str) else tuple(x),
        getter_fct=lambda x: sorted(x)
    )

    #: str: Tag abbr
    tags = TProp(
        Union[None, Iterable[str]],
        setter_fct=lambda x: set(x) if x else set(),
        getter_fct=lambda x: sorted(x)
    )

    def __init__(self, evt_dt, srn, tags=None):
        """Constructor

        Args:
            evt_dt (str | datetime | pd.Timestamp): UTC datetime
            srn (str): Instrument serial number
            tags (`optional`, iterable of str): Tags

        """

        # Set attributes
        self.evt_dt = evt_dt
        self.srn = srn
        self.tags = tags

    def __copy__(self):
        return self.__class__(self.evt_dt, self.srn.copy(), self.tags.copy())

    @property
    def evt_id(self):
        """str: Event ID which match 1st corresponding pattern in tags. Defaults to None."""
        try:
            out = next(filter(glob_var.evt_id_pat.match, self.tags))
        except StopIteration:
            out = None
        return out

    @property
    def rig_id(self):
        """str: Rig ID which match 1st corresponding pattern in tags. Defaults to None."""
        try:
            out = next(filter(glob_var.rig_id_pat.match, self.tags))
        except StopIteration:
            out = None
        return out

    @property
    def mdl_id(self):
        """str: GDP model ID which match 1st corresponding pattern in tags. Defaults to None."""
        try:
            out = next(filter(glob_var.mdl_id_pat.match, self.tags))
        except StopIteration:
            out = None
        return out

    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        p_printer = pprint.PrettyPrinter()
        return p_printer.pformat(
            (f'dt: {self.evt_dt}', f'srn: {self.srn}', f'tag: {self.tags}')
        )

    def __hash__(self):
        return hash(self.sort_attr)

    def add_tag(self, val):
        """Add a tag abbr

        Args:
            val (list of str): tag abbr to add

        """

        if isinstance(val, str):
            # Assume the user forgot to put the key into a list.
            val = [val]

        # Add
        self.tags = self.tags + val

    def rm_tag(self, val):
        """Remove a tag abbr

        Args:
            val (list of str): tag abbr to remove

        """

        if isinstance(val, str):
            # Assume the user forgot to put the key into a list.
            val = [val]

        # Remove
        self.tags = list(filter(lambda x: x not in val, self.tags))

    @staticmethod
    def sort(info_list):
        """Sort list of InfoManager. Sorting order [evt_dt, srn, tags]

        Args:
            info_list (list of InfoManager): List to sort

        Returns:
            list: Sorted InfoManager
            tuple: Original list index

        """

        # Sort
        val = sorted(
            zip(info_list, range(len(info_list)))
        )

        # Unzip index
        out = list(unzip(val)) if val else ([], [])

        return list(out[0]), out[1]

    @property
    def sort_attr(self):
        """ list of InfoManger attributes: Attributes sort order"""
        return tuple((self.evt_dt, *self.srn, *self.tags))

    def __eq__(self, other):
        return self.sort_attr == other.sort_attr

    def __ne__(self, other):
        return self.sort_attr != other.sort_attr

    def __lt__(self, other):
        return self.sort_attr < other.sort_attr

    def __le__(self, other):
        return self.sort_attr <= other.sort_attr

    def __gt__(self, other):
        return self.sort_attr > other.sort_attr

    def __ge__(self, other):
        return self.sort_attr >= other.sort_attr


class SearchInfoExpr(metaclass=ABCMeta):
    """Abstract search info expression interpreter class.

    .. uml::

        @startuml
        footer Interpreter design pattern

        class SearchInfoExpr {
            {abstract} interpret()
            {static} eval()
        }

        class LogicalSearchInfoExpr {
            _expression: List
            interpret()
            {abstract} fct(*arg)
        }

        SearchInfoExpr <|-- LogicalSearchInfoExpr : extends
        LogicalSearchInfoExpr o--> SearchInfoExpr

        class TerminalSearchInfoExpr {
            interpret()
            {abstract} get_filter()
        }

        SearchInfoExpr <|-- TerminalSearchInfoExpr : extends

        @enduml

    """

    @abstractmethod
    def interpret(self):
        """Interpreter method"""

    @staticmethod
    def eval(str_expr, prm_abbr, filter_empty):
        """Evaluate search expression

        Args:
            str_expr (str): Expression to evaluate
            prm_abbr (str): Search parameter
            filter_empty (bool): Filter for empty data

        Returns:
            List of Info.id

        Search expression grammar:
            - all(): Select all
            - [datetime ; dt]('<ISO datetime>', ['=='(default) ; '>=' ; '>' ; '<=' ; '<' ; '!=']): Select by datetime
            - [serialnumber ; srn]('<Serial number>'): Select by serial number
            - tag(['<Tag>' ; ('<Tag 1>', ...,'<Tag n>')]): Select by tag
            - and_(<expr 1>, ..., <expr n>): Intersection
            - or_(<expr 1>, ..., <expr n>): Union
            - not_(<expr>): Negation, correspond to all() without <expr>
        """

        # Define
        str_expr_dict = {
            'all': AllExpr,
            'datetime': DatetimeExpr, 'dt': DatetimeExpr,
            'serialnumber': SerialNumberExpr, 'srn': SerialNumberExpr,
            'tag': TagExpr,
            'and_': AndExpr,
            'or_': OrExpr,
            'not_': NotExpr
        }
        db_mngr = DatabaseManager()

        with DBAccess(db_mngr) as _:

            # Eval expression
            expr = eval(str_expr, str_expr_dict)

            # Add empty tag if False
            if filter_empty is True:
                expr = AndExpr(NotExpr(TagExpr(TAG_EMPTY_VAL)), expr)

            # Filter parameter
            expr = AndExpr(ParameterExpr(prm_abbr), expr)

            # Interpret expression
            expr_res = expr.interpret()

            # Convert id as table element
            qry = Info.select().where(Info.id.in_(expr_res))
            out = [arg for arg in qry.iterator()]

            return out


class LogicalSearchInfoExpr(SearchInfoExpr):
    """
    Implement an interpret operation for nonterminal symbols in the grammar.
    """

    def __init__(self, *args):
        self._expression = args

    def interpret(self):
        """Terminal interpreter method"""
        return reduce(
            self.fct,
            [arg.interpret() for arg in self._expression]
        )

    @abstractmethod
    def fct(self, *args):
        """Logical function between expression args"""


class AndExpr(LogicalSearchInfoExpr):
    """And operation"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.and_(a, b)


class OrExpr(LogicalSearchInfoExpr):
    """Or operation"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.or_(a, b)


class NotExpr(LogicalSearchInfoExpr):
    """Not operation"""

    def __init__(self, arg):
        self._expression = [AllExpr(), arg]

    def fct(self, a, b):
        """Implement fct method"""
        return operator.sub(a, b)


class TerminalSearchInfoExpr(SearchInfoExpr):
    """
    Implement an interpret operation associated with terminal symbols in
    the grammar.
    """

    QRY_BASE = (
        Info
        .select().distinct()
        .join(InfosInstruments).join(Instrument).switch(Info)
        .join(Parameter).switch(Info)
        .join(InfosTags).join(Tag).switch(Info)
    )

    def __init__(self, arg):
        self.expression = arg

    def interpret(self):
        """Terminal expression interpreter"""
        return set(
            arg.id for arg in
            self.QRY_BASE.where(self.get_filter()).iterator()
        )

    @abstractmethod
    def get_filter(self):
        """Return query where method filter"""


class AllExpr(TerminalSearchInfoExpr):
    """All filter"""

    def __init__(self):
        pass

    def get_filter(self):
        """Implement get_filter method"""
        return


class DatetimeExpr(TerminalSearchInfoExpr):
    """Datetime filter"""

    _OPER_DICT = {
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '>=': operator.le,
    }
    expression = TProp(Union[str, Timestamp, datetime], check_datetime)

    def __init__(self, arg, op='=='):
        self.expression = arg
        self._op = self._OPER_DICT[op]

    def get_filter(self):
        """Implement get_filter method"""
        return self._op(Info.evt_dt, self.expression)


class SerialNumberExpr(TerminalSearchInfoExpr):
    """Serial number filter"""

    expression = TProp(str, lambda *x: x[0])

    def get_filter(self):
        """Implement get_filter method"""
        return Instrument.srn == self.expression


class TagExpr(TerminalSearchInfoExpr):
    """Tag filter"""

    expression = TProp(
        Union[str, Iterable[str]], lambda x: set([x]) if isinstance(x, str) else set(x)
    )

    def get_filter(self):
        """Implement get_filter method"""
        return Tag.tag_txt.in_(self.expression)


class ParameterExpr(TerminalSearchInfoExpr):
    """Parameter filter"""

    expression = TProp(str, lambda x: x)

    def get_filter(self):
        """Implement get_filter method"""
        return Parameter.prm_abbr == self.expression


class DBCreateError(Exception):
    """Exception class for DB creation error"""


class DBInsertError(Exception):
    """Exception class for DB insert error"""


class DBDirError(Exception):
    """Exception class for DB directory creating error"""


class ConfigGenMaxLenError(Exception):
    """Exception class for max length config generator error"""
