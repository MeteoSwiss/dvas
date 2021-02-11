"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Local database management tools

"""

# Import from python packages
import pprint
from pathlib import Path
from hashlib import blake2b
from abc import abstractmethod, ABCMeta
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


# Import from current package
from .model import db
from .model import InstrType as TableInstrType
from .model import Object as TableObject
from .model import Info as TableInfo
from .model import Parameter as TableParameter
from .model import Tag as TableTag
from .model import MetaData as TableMetaData
from .model import Flag, DataSource, Data
from .model import InfosObjects as TableInfosObjects
from .model import InfosTags
from ..config.config import OneDimArrayConfigLinker
from ..config.definitions.origdata import EVT_DT_FLD_NM
from ..config.definitions.origdata import TAG_FLD_NM, META_FLD_NM
from ..config.definitions.tag import TAG_NONE, TAG_EMPTY_VAL
from ..helper import ContextDecorator
from ..helper import SingleInstanceMetaClass
from ..helper import TypedProperty as TProp
from ..helper import TimeIt
from ..helper import get_by_path, check_datetime
from ..helper import unzip, get_dict_len
from ..logger import localdb
from ..environ import glob_var
from ..environ import path_var as env_path_var

# Define
SQLITE_MAX_VARIABLE_NUMBER = 999
INDEX_NM = Data.index.name
VALUE_NM = Data.value.name

#: int: Database cache size in kB
DB_CACHE_SIZE = 10 * 1024

#: str: Local database file name
DB_FILE_NM = 'local_db.sqlite'


class DatabaseManager(metaclass=SingleInstanceMetaClass):
    """Local data base manager.

    Note:
        If the data base does not exists, the creation will be forced.

    """

    DB_TABLES = [
        TableInfo,
        TableInfosObjects, TableObject, TableInstrType,
        InfosTags, TableTag,
        DataSource,
        Data,
        TableMetaData,
        TableParameter,
        Flag,
    ]
    DB_TABLES_PRINT = [
        TableParameter, TableInstrType,
        TableObject, Flag,
        TableTag
    ]

    @staticmethod
    def clear_db():
        """Clear DB

        Note:
            !!!ONLY FOR ADVANCED USER!!! Use this method carefully.
            Ensure that no more class instance are linked to this reference.

        """

        # Get db file path
        db_mngr = DatabaseManager()
        db_file_path = db_mngr.db.database

        # Delete singleton instance
        del db_mngr
        SingleInstanceMetaClass.pop_instance(DatabaseManager)

        # Delete file
        Path(db_file_path).unlink()

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
                raise DBDirError(f"Error in creating '{self._db.database.parent}' ({exc})") from exc

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
    def model_to_dict(query, recurse=False):
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
                for tbl in [TableParameter, TableInstrType, Flag, TableTag]:
                    self._fill_table(tbl)

            except IntegrityError as exc:
                raise DBCreateError(exc) from exc

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

    def get_table(self, table, search=None, recurse=False):
        """

        Args:
            table:
            search (dict): key 'join_order' must be a list of database.database.MetadataModel, `optional`,
                key 'where' a logical peewee expression

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
            out = self.model_to_dict(qry, recurse=recurse)

        return out

    def add_data(
            self, index, value, info, prm_name,
            source_info=None, force_write=False
    ):
        """Add profile data to the DB.

        Args:
            index (np.array of int): Data index
            value (np.array of float): Data value
            info (InfoManager|dict): Data information. If dict, must fulfill
                InfoManager.from_dict input args requirements.
            prm_name (str):
            source_info (str, optional): Data source
            force_write (bool, optional): force rewrite of already save data

        Raises:
            DBInsertError: Error while add data

        """

        # Test input
        try:
            assert len(index) == len(value),\
                "Data index and data value are of different length"
            assert isinstance(index, np.ndarray), "Data index is not an np.ndarray"
            assert isinstance(value, np.ndarray), "Data value is not an np.ndarray"
        except AssertionError as ass:
            raise DBInsertError(ass)

        # Convert to InfoManager
        if isinstance(info, dict):
            info = InfoManager.from_dict(info)

        # Add data
        try:
            with DBAccess(self) as _:

                # Check instrument id existence
                if (
                    oid_list := sorted([
                        arg[0] for arg in
                        self.get_or_none(
                            TableObject,
                            search={
                                'where': TableObject.oid.in_(info.oid)
                            },
                            attr=[[TableObject.oid.name]],
                            get_first=False
                        )
                    ])
                ) != info.oid:
                    err_msg = f"Many instrument id in %s are missing in DB"
                    localdb.error(err_msg, info.oid)
                    raise DBInsertError(err_msg % info.oid)

                # Get/Check parameter
                param = TableParameter.get_or_none(
                    TableParameter.prm_name == prm_name
                )
                if not param:
                    err_msg = "prm_name '%s' is missing in DB"
                    localdb.error(err_msg, prm_name)
                    raise DBInsertError(err_msg % prm_name)

                # Check tag_name existence
                if len(
                    tags_id_list := [
                        arg[0] for arg in
                        self.get_or_none(
                            TableTag,
                            search={
                                'where': TableTag.tag_name.in_(info.tags)
                            },
                            attr=[[TableTag.id.name]],
                            get_first=False
                        )
                    ]
                ) != len(info.tags):
                    err_msg = "Many tags in %s are missing in DB"
                    localdb.error(err_msg, info.tags)
                    raise DBInsertError(err_msg % info.tags)

                # Create original data information
                data_src, _ = DataSource.get_or_create(source=source_info)

                # Create info
                info_id, created = TableInfo.get_or_create(
                    evt_dt=info.evt_dt, param=param,
                    data_src=data_src, evt_hash=info.get_hash()
                )

                # Erase data (created == False indicate that data already exists)
                if (created is False) and (force_write is True):

                    # Delete InfosTags entries
                    InfosTags.delete().\
                        where(InfosTags.info == info_id).\
                        execute()

                    # Delete TableInfosObjects entries
                    TableInfosObjects.delete().\
                        where(TableInfosObjects.info == info_id).\
                        execute()

                    # Delete Data entries
                    Data.delete().\
                        where(Data.info == info_id).\
                        execute()

                    # Delete Metadata entries
                    TableMetaData.delete().\
                        where(TableMetaData.info == info_id).\
                        execute()

                    # TODO
                    #  Add log

                # Insert data (created == True indicate that data are new)
                if (created is True) or (force_write is True):

                    # Link info to tag
                    tags_info = [
                        {
                            InfosTags.tag.name: tag_id,
                            InfosTags.info.name: info_id
                        } for tag_id in tags_id_list
                    ]
                    if tags_info:

                        # Calculate max batch size
                        n_max = floor(SQLITE_MAX_VARIABLE_NUMBER/get_dict_len(tags_info[0]))

                        # Insert
                        for batch in chunked(tags_info, n_max):
                            InfosTags.insert_many(batch).execute()  # noqa pylint: disable=E1120

                    # Link info to instrument
                    object_info = [
                        {
                            TableInfosObjects.object.name: oid,
                            TableInfosObjects.info.name: info_id
                        } for oid in oid_list
                    ]
                    if object_info:

                        # Calculate max batch size
                        n_max = floor(SQLITE_MAX_VARIABLE_NUMBER / get_dict_len(object_info[0]))

                        # Insert
                        for batch in chunked(object_info, n_max):
                            TableInfosObjects.insert_many(batch).execute()  # noqa pylint: disable=E1120

                    # Add metadata
                    # ------------

                    # Create batch index
                    fields = [
                        TableMetaData.key_name, TableMetaData.value_str,
                        TableMetaData.value_num, TableMetaData.info
                    ]

                    # Create batch data
                    batch_data = [
                        (key,
                         val if isinstance(val, str) else None,
                         val if isinstance(val, float) else None,
                         info_id)
                        for key, val in info.metadata.items()
                    ]

                    # Calculate max batch size
                    n_max = floor(SQLITE_MAX_VARIABLE_NUMBER / len(fields))

                    # Insert to db
                    for batch in chunked(batch_data, n_max):
                        TableMetaData.insert_many(batch, fields=fields).execute()  # noqa pylint: disable=E1120

                    # Add Data
                    # --------

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

                else:

                    # TODO
                    #  Add log

                    pass

        except DBInsertError as exc:
            raise DBInsertError(exc)

    @staticmethod
    def _get_info_id(search_expr, prm_name, filter_empty):
        """Get Info.info_id for a give search string expression

        Args:
            search_expr (str): Search expression
            prm_name (str): Parameter name
            filter_empty (bool): Filter for empty data tag

        """

        try:
            out = list(SearchInfoExpr.eval(search_expr, prm_name, filter_empty))

        # TODO Detail exception
        except Exception as exc:
            print(f'Error in search expression {search_expr} ({exc})')

            # TODO Decide if raise or not
            out = []

        return out

    @TimeIt()
    def get_data(self, search_expr, prm_name, filter_empty):
        """Get data from DB

        Args:
            search_expr (str): Search expression
            prm_name (str): Parameter name
            filter_empty (bool): Filter empty data or not

        Returns:

        """

        # Get info id
        info_id_list = self._get_info_id(search_expr, prm_name, filter_empty)

        if not info_id_list:
            localdb.warning(
                "Empty search '%s' for '%s", search_expr, prm_name
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

                # Get related instrument id
                oid_list = [
                    arg.oid for arg in
                    TableObject.select().distinct().
                        join(TableInfosObjects).join(TableInfo).
                        where(TableInfo.info_id == info_id_list[i].info_id).
                        iterator()
                ]

                # Get related tags
                tag_name_list = [
                    arg.tag_name for arg in
                    TableTag.select().distinct().
                    join(InfosTags).join(TableInfo).
                    where(TableInfo.info_id == info_id_list[i].info_id).
                    iterator()
                ]

                # Get related metadata
                metadata_dict = {
                    arg.key_name: arg.value_num if arg.value_str is None else arg.value_str
                    for arg in
                    TableMetaData.select().distinct().
                    join(TableInfo).
                    where(TableInfo.info_id == info_id_list[i].info_id).
                    iterator()
                }

                # Append
                out.append(
                    {
                        'info': InfoManager(
                            evt_dt=info_id_list[i].evt_dt,
                            oid=oid_list,
                            tags=tag_name_list,
                            metadata=metadata_dict,
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
        out = f"\nDatabase content in '{self.db.database}:'\n"
        out += f"{'*' * len(out)}\n"

        if not print_tables:
            print_tables = self.DB_TABLES_PRINT

        for print_tbl in print_tables:
            out += f"{print_tbl.__name__}\n"
            for arg in self.get_table(print_tbl, recurse=recurse):
                out += f"{arg}\n"
            out += "\n"

        return out

    def get_flags(self):
        """Get config flags

        Returns:
            list

        """
        return self.get_table(Flag)


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


class InfoManagerMetaData(dict):
    """Class to define metadata allowed types

    Note:
        This class is used to bypass the missing class Mapping in
        pampy package.

    """

    def __init__(self, dict_args={}):
        """
        Args:
            dict_args (dict): keys must be str and values (str, float)
        """
        self._check_and_convert(dict_args)
        super().__init__(**dict_args)

    def copy(self):
        return InfoManagerMetaData(super().copy())

    def update(self, dict_args):
        """Update dict

        Args:
            dict_args (dict): keys must be str and values (str, float)

        """
        self._check_and_convert(dict_args)
        super().update(dict_args)

    @staticmethod
    def _check_and_convert(dict_args):
        """Method to check dict key and value types.

        Note:
            Type int will be converted to float.

        Args:
            dict_args (dict): keys must be str and values (str, float)
        """

        # Check
        try:
            assert isinstance(dict_args, dict)
            assert all([isinstance(key, str) for key in dict_args.keys()])
            assert all([isinstance(val, (str, float, int)) for val in dict_args.values()])
        except AssertionError:
            raise TypeError()

        # Convert
        for key, val in dict_args.items():
            if isinstance(val, int):
                dict_args.update({key: float(val)})


class InfoManager:
    """Data info manager"""

    #: datetime.datetime: UTC datetime
    evt_dt = TProp(Union[str, Timestamp, datetime], check_datetime)

    #: int|iterable of int: Object id
    oid = TProp(
        Union[int, Iterable[int]],
        setter_fct=lambda x: (x,) if isinstance(x, int) else tuple(x),
        getter_fct=lambda x: sorted(x)
    )

    #: str|iterable of str: Tags
    tags = TProp(
        Union[str, Iterable[str]],
        setter_fct=lambda x: set((x,)) if isinstance(x, str) else set(x),
        getter_fct=lambda x: sorted(x)
    )

    #: dict: Metadata
    metadata = TProp(InfoManagerMetaData, getter_fct= lambda x: x.copy())

    def __init__(self, evt_dt, oid, tags=TAG_NONE, metadata={}):
        """Constructor

        Args:
            evt_dt (str | datetime | pd.Timestamp): UTC datetime
            oid (int|iterable of int): Object identifier (snr, pid)
            tags (str|iterable of str, `optional`): Tags. Defaults to ''
            metadata (dict|InfoManagerMetaData, `optional`): Default to {}

        """

        # Init
        if isinstance(metadata, dict):
            metadata = InfoManagerMetaData(metadata)

        # Set attributes
        self.evt_dt = evt_dt
        self.oid = oid
        self.tags = tags
        self.metadata = metadata

    def __copy__(self):
        return self.__class__(self.evt_dt, self.oid.copy(), self.tags.copy())

    @property
    def evt_id(self):
        """str: Event ID which match 1st corresponding pattern in tags. Defaults to None."""
        try:
            # TODO: the following line triggers a *very* weird pylint Error 1101.
            # I disable it for now ... but someone should really confirm whether this ok or not!
            # fpavogt - 2020.12.09
            out = next(filter(glob_var.evt_id_pat.match, self.tags)) # pylint: disable=E1101
        except StopIteration:
            out = None
        return out

    @property
    def rig_id(self):
        """str: Rig ID which match 1st corresponding pattern in tags. Defaults to None."""
        try:
            # TODO: the following line triggers a *very* weird pylint Error 1101.
            # I disable it for now ... but someone should really confirm whether this ok or not!
            # fpavogt - 2020.12.09
            out = next(filter(glob_var.rig_id_pat.match, self.tags))  # pylint: disable=E1101
        except StopIteration:
            out = None
        return out

    @property
    def mdl_id(self):
        """str: GDP model ID which match 1st corresponding pattern in tags. Defaults to None."""
        try:
            # TODO: the following line triggers a *very* weird pylint Error 1101.
            # I disable it for now ... but someone should really confirm whether this ok or not!
            # fpavogt - 2020.12.09
            out = next(filter(glob_var.mdl_id_pat.match, self.tags)) # pylint: disable=E1101
        except StopIteration:
            out = None
        return out

    @property
    def tags_desc(self):
        """dict: Tags description"""

        # Define
        db_mngr = DatabaseManager()

        # Query for tags informations
        qry_res = db_mngr.get_table(
            TableTag,
            search={'where': TableTag.tag_name.in_(self.tags)}
        )

        # Set output
        out = {
            res[TableTag.tag_name.name]: res[TableTag.tag_desc.name]
            for res in qry_res
            }

        return out

    @property
    def object(self):
        """list of dict: Object details"""

        # Define
        db_mngr = DatabaseManager()

        # Query for instrument informations
        qry_res = db_mngr.get_table(
            TableObject,
            search={
                'join_order': [TableInstrType],
                'where': TableObject.oid.in_(self.oid)
            },
            recurse=True
        )

        # Set output
        out = [
            {
                TableObject.oid.name: res[TableObject.oid.name],
                TableObject.srn.name: res[TableObject.srn.name],
                TableObject.pid.name: res[TableObject.pid.name],
                TableInstrType.type_name.name: res[TableObject.instr_type.name][TableInstrType.type_name.name],
                TableInstrType.type_desc.name: res[TableObject.instr_type.name][TableInstrType.type_desc.name]
            }
            for res in qry_res
        ]

        return out

    def __repr__(self):
        return f'{self}'

    def __str__(self):
        p_printer = pprint.PrettyPrinter()
        return p_printer.pformat(
            (f'evt_dt: {self.evt_dt}', f'oid: {self.oid}',
             f'tags: {self.tags}', f'metadata: {self.metadata}')
        )

    def get_hash(self):
        """Return 20 bytes hash as string"""
        return blake2b(
            b''.join([str(arg).encode('utf-8') for arg in self._get_attr_sort_order()]),
            digest_size=20
        ).hexdigest()

    def __hash__(self):
        return hash(self._get_attr_sort_order())

    def add_tags(self, val):
        """Add a tag name

        Args:
            val (list of str): Tag names to add

        """

        if isinstance(val, str):
            # Assume the user forgot to put the key into a list.
            val = [val]

        # Add
        self.tags = self.tags + val

    def rm_tags(self, val):
        """Remove a tag name

        Args:
            val (list of str): Tag names to remove

        """

        if isinstance(val, str):
            # Assume the user forgot to put the key into a list.
            val = [val]

        # Remove
        self.tags = list(filter(lambda x: x not in val, self.tags))

    def add_metadata(self, key, val):
        """Add metadata

        Args:
            key (str): Metadata key
            val (str, float, int, bool): Associated value

        """
        metadata = self.metadata
        metadata.update({key: val})
        self.metadata = metadata

    def rm_metadata(self, key):
        """Remove metadata

        Args:
            key (str): Metadata key to be removed

        """
        metadata = self.metadata
        metadata.pop(key)
        self.metadata = metadata

    @staticmethod
    def sort(info_list):
        """Sort list of InfoManager. Sorting order [evt_dt, srn, tags]

        Args:
            info_list (iterable of InfoManager): List to sort

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

    def _get_attr_sort_order(self):
        """Return attributes used for sorting/hashing

        Returns:
            tuple

        """
        return self.evt_dt, *[str(arg) for arg in self.oid], *self.tags

    def __eq__(self, other):
        return self._get_attr_sort_order() == other._get_attr_sort_order()

    def __ne__(self, other):
        return self._get_attr_sort_order() != other._get_attr_sort_order()

    def __lt__(self, other):
        return self._get_attr_sort_order() < other._get_attr_sort_order()

    def __le__(self, other):
        return self._get_attr_sort_order() <= other._get_attr_sort_order()

    def __gt__(self, other):
        return self._get_attr_sort_order() > other._get_attr_sort_order()

    def __ge__(self, other):
        return self._get_attr_sort_order() >= other._get_attr_sort_order()

    @staticmethod
    def from_dict(metadata):
        """Convert dict of metadata to InfoManager

        Dict keys:
            - evt_dt (str): Datetime
            - typ_name (str, `optional`): Instrument type (used to create
                instrument entry if missing in DB)
            - srn_field (str): Serial number
            - pid (str): Product identifier
            - tags (list of str): Tags
            - meta_field (dict): Metadata as dict

        """

        # Define
        db_mngr = DatabaseManager()

        # Get instrument id
        if (
            oid := db_mngr.get_or_none(
                TableObject,
                search={
                    'where': (
                        (TableObject.srn == metadata[TableObject.srn.name]) &
                        (TableObject.pid == metadata[TableObject.pid.name])
                    )
                },
                attr=[[TableObject.oid.name]]
            )
        ) is None:

            # Get instrument type
            if (
                instr_type := db_mngr.get_or_none(
                    TableInstrType,
                    search={
                        'where': TableInstrType.type_name == metadata[TableInstrType.type_name.name]
                    }
                )
            ) is None:
                # TODO
                #  Detail exception
                raise Exception(f"{metadata[TableInstrType.type_name.name]} is missing in DB/InstrumentType")

            # Create instrument entry
            with DBAccess(db_mngr):
                oid = TableObject.create(
                    srn=metadata[TableObject.srn.name],
                    pid=metadata[TableObject.pid.name],
                    instr_type=instr_type
                ).oid

        # Construct InfoManager
        try:
            info = InfoManager(
                evt_dt=metadata[EVT_DT_FLD_NM],
                oid=oid,
                tags=metadata[TAG_FLD_NM],
                metadata=metadata[META_FLD_NM]
            )
        except Exception as exc:
            # TODO
            #  Detail exception
            raise Exception(exc)

        return info


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
    def eval(str_expr, prm_name, filter_empty):
        """Evaluate search expression

        Args:
            str_expr (str): Expression to evaluate
            prm_name (str): Search parameter
            filter_empty (bool): Filter for empty data

        Returns:
            List of Info.info_id

        Search expression grammar:
            - all(): Select all
            - [datetime ; dt]('<ISO datetime>', ['=='(default) ; '>=' ; '>' ; '<=' ; '<' ; '!=']): Select by datetime
            - [serialnumber ; srn]('<Serial number>'): Select by serial number
            - [product_id ; pid](<Product>): Select by product
            - tags(['<Tag>' ; ('<Tag 1>', ...,'<Tag n>')]): Select by tag
            - and_(<expr 1>, ..., <expr n>): Intersection
            - or_(<expr 1>, ..., <expr n>): Union
            - not_(<expr>): Negation, correspond to all() without <expr>
        """

        # Define
        str_expr_dict = {
            'all': AllExpr,
            'datetime': DatetimeExpr, 'dt': DatetimeExpr,
            'serialnumber': SerialNumberExpr, 'srn': SerialNumberExpr,
            'product_id': ProductExpr, 'pid': ProductExpr,
            'tags': TagExpr,
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
            expr = AndExpr(ParameterExpr(prm_name), expr)

            # Interpret expression
            expr_res = expr.interpret()

            # Convert id as table element
            qry = TableInfo.select().where(TableInfo.info_id.in_(expr_res))
            out = [arg for arg in qry.iterator()]

            # TODO
            #  Raise exception

            return out


class LogicalSearchInfoExpr(SearchInfoExpr):
    """
    Implement an interpret operation for nonterminal symbols in the grammar.
    """

    def __init__(self, *args):
        self._expression = args

    def interpret(self):
        """Non terminal interpreter method"""
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
        TableInfo
        .select().distinct()
        .join(TableInfosObjects).join(TableObject).switch(TableInfo)
        .join(TableParameter).switch(TableInfo)
        .join(InfosTags).join(TableTag).switch(TableInfo)
    )

    def __init__(self, arg):
        self.expression = arg

    def interpret(self):
        """Terminal expression interpreter"""
        return set(
            arg.info_id for arg in
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
        return self._op(TableInfo.evt_dt, self.expression)


class SerialNumberExpr(TerminalSearchInfoExpr):
    """Serial number filter"""

    expression = TProp(
        Union[str, Iterable[str]],
        setter_fct=lambda x: [x] if isinstance(x, str) else list(x)
    )

    def get_filter(self):
        """Implement get_filter method"""
        return TableObject.srn.in_(self.expression)


class ProductExpr(TerminalSearchInfoExpr):
    """Product filter"""

    expression = TProp(
        Union[int, Iterable[int]],
        setter_fct=lambda x: [x] if isinstance(x, int) else list(x)
    )

    def get_filter(self):
        """Implement get_filter method"""
        return TableObject.pid.in_(self.expression)


class TagExpr(TerminalSearchInfoExpr):
    """Tag filter"""

    expression = TProp(
        Union[str, Iterable[str]], lambda x: set((x,)) if isinstance(x, str) else set(x)
    )

    def get_filter(self):
        """Implement get_filter method"""
        return TableTag.tag_name.in_(self.expression)


class ParameterExpr(TerminalSearchInfoExpr):
    """Parameter filter"""

    expression = TProp(str, lambda x: x)

    def get_filter(self):
        """Implement get_filter method"""
        return TableParameter.prm_name == self.expression


class DBCreateError(Exception):
    """Exception class for DB creation error"""


class DBInsertError(Exception):
    """Exception class for DB insert error"""


class DBDirError(Exception):
    """Exception class for DB directory creating error"""
