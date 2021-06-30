"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Local database management tools

"""

# Import from python packages
import pprint
from hashlib import blake2b
from math import floor
from datetime import datetime
from peewee import chunked, DoesNotExist
from playhouse.shortcuts import model_to_dict
import numpy as np
from pandas import Timestamp
from pampy.helpers import Iterable, Union

# Import from current package
from .model import db
from .model import Model as TableModel
from .model import Object as TableObject
from .model import Info as TableInfo
from .model import Parameter as TableParameter
from .model import Tag as TableTag
from .model import MetaData as TableMetaData
from .model import Flg, DataSource, Data
from .model import InfosObjects as TableInfosObjects
from .model import InfosTags
from .search import SearchInfoExpr
from ..config.config import Parameter as ParameterCfg
from ..config.config import Model as ModelCfg
from ..config.config import Flg as FlgCfg
from ..config.config import Tag as TagCfg
from ..config.config import instantiate_config_managers
from ..config.definitions.origdata import EDT_FLD_NM
from ..config.definitions.origdata import TAG_FLD_NM, META_FLD_NM
from ..hardcoded import TAG_NONE_NAME
from ..helper import SingleInstanceMetaClass
from ..helper import TypedProperty as TProp
from ..helper import get_by_path, check_datetime
from ..helper import unzip, get_dict_len
from ..logger import localdb as localdb_logger
from ..logger import log_func_call
from ..environ import glob_var
from ..environ import path_var as env_path_var
from ..errors import DvasError
from .. import dynamic as dyn

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
        TableInfosObjects, TableObject, TableModel,
        InfosTags, TableTag,
        DataSource,
        Data,
        TableMetaData,
        TableParameter,
        Flg,
    ]

    def __init__(self):

        # Create config linker instance attribute
        self._cfg_mngr = instantiate_config_managers(ParameterCfg, ModelCfg, FlgCfg, TagCfg,
                                                     read=True)

        # Create db attribute
        self._db = db

        # Init db
        db_new = self._init_db()

        # Create table if new
        if db_new:
            self._create_tables()
            self._fill_metadata()

    @property
    def db(self):
        """peewee.SqliteDatabase: Database instance"""
        return self._db

    def refresh_db(self):
        """ Refreshes the database, by deleting the current tables and reloading them with fresh
        metadata. """
        self._delete_tables()
        self._create_tables()
        self._fill_metadata()

    def _init_db(self):
        """ Init db. Create new file if missing or take existing file.

        Returns:
            bool: True if the DB is newly created, and not in memory.

        """

        # Test if I got a proper path to store the DB (unless I was asked to store it in memory).
        if env_path_var.local_db_path is None and not dyn.DB_IN_MEMORY:
            raise DvasError("Ouch ! I can't find any value for "+
                            "`dvas.environ.path_var.local_db_path`. Was it properly defined ?")

        # Define the DB parameters
        pragmas = {'foreign_keys': True,
                   'cache_size': -DB_CACHE_SIZE,  # Set cache to 10MB
                   'permanent': True,
                   'synchronous': False,
                   'journal_mode': 'MEMORY'
                  }

        if dyn.DB_IN_MEMORY:
            file_path = ':memory:'
            # The following assumes that in case of in-memory storage, the db is always new.
            # Assumed to be correct, until proven otherwise. fpavogt, 14.06.2021
            db_new = True

        else:
            file_path = env_path_var.local_db_path / DB_FILE_NM

            # Create local DB directory
            if file_path.exists():
                db_new = False

            else:
                db_new = True
                try:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    # Set user read/write permission
                    file_path.parent.chmod(file_path.parent.stat().st_mode | 0o600)
                except (OSError,) as exc:
                    raise DBDirError(f"Error in creating '{self._db.database.parent}' ({exc})") \
                        from exc

        # Ready to actually initialize the DB
        self._db.init(file_path, pragmas=pragmas)
        return db_new

    @staticmethod
    def get_or_none(table, search=None, attr=None, get_first=True):
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
        for table in self.DB_TABLES:
            table.create_table(safe=True)

    def _delete_tables(self):
        """Delete table instances"""
        for table in self.DB_TABLES:
            qry = table.delete()
            qry.execute()  # noqa pylint: disable=E1120

    def _fill_metadata(self):
        """Create db tables"""

        # Fill simple tables
        for tbl in [TableParameter, TableModel, Flg, TableTag]:
            self._fill_table(tbl)

    def _fill_table(self, table, foreign_constraint=None):
        """

        Args:
            table:
            foreign_constraint:

        """

        # Init
        document = self._cfg_mngr[table.__name__]

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
            search (dict, optional): key 'join_order' must be a list of database.database.MetadataModel,
                `optional`, key 'where' a logical peewee expression

        Returns:
            dict:

        """

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
            self, index, value, info, prm_name, force_write=False
    ):
        """Add profile data to the DB.

        Args:
            index (np.array of int): Data index
            value (np.array of float): Data value
            info (InfoManager|dict): Data information. If dict, must fulfill
                InfoManager.from_dict input args requirements.
            prm_name (str):
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
                localdb_logger.error(err_msg, info.oid)
                raise DBInsertError(err_msg % info.oid)

            # Get/Check parameter
            if not (param := TableParameter.get_or_none(
                TableParameter.prm_name == prm_name)
            ):
                err_msg = "prm_name '%s' is missing in DB"
                localdb_logger.error(err_msg, prm_name)
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
                localdb_logger.error(err_msg, info.tags)
                raise DBInsertError(err_msg % info.tags)

            # Create original data information
            data_src, _ = DataSource.get_or_create(src=info.src)

            # Create info
            info_id, created = TableInfo.get_or_create(
                edt=info.edt, param=param,
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
                # TODO
                #  Test multithreading in order to speed up data select
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

        # Init
        SearchInfoExpr.set_stgy('info')

        try:
            out = list(SearchInfoExpr.eval(search_expr, prm_name=prm_name, filter_empty=filter_empty))

        # TODO Detail exception
        except Exception as exc:
            print(f'Error in search expression {search_expr} ({exc})')

            # TODO Decide if raise or not
            out = []

        return out

    @log_func_call(localdb_logger, time_it=True)
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
            localdb_logger.warning(
                "Empty search '%s' for '%s", search_expr, prm_name
            )

        # Query data
        res = []
        for info_id in info_id_list:

            # TODO
            #  Consider to implement multithreading in order to speed up data select
            try:
                qry = (
                    Data.
                    select(Data.index, Data.value).
                    where(Data.info == info_id)
                )

                # 0: index, 1: value
                res.append(tuple(unzip(qry.tuples().iterator())))

            # TODO
            #  Detail exception
            except Exception as ex:
                raise Exception(ex)

        # Group data
        out = []
        for i, arg in enumerate(res):

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

            # Get source
            if not (data_src := [arg.src for arg in
                    DataSource.select().distinct().
                    join(TableInfo).
                    where(TableInfo.data_src == DataSource.id).
                    where(TableInfo.info_id == info_id_list[i].info_id).
                    iterator()
                ]
            ):
                # TODO
                #  Detail exception
                raise Exception(f'Data source is empty')

            # Append
            out.append(
                {
                    'info': InfoManager(
                        edt=info_id_list[i].edt,
                        oid=oid_list,
                        tags=tag_name_list,
                        metadata=metadata_dict,
                        src=data_src[0]
                    ),
                    'index': arg[0],
                    'value': arg[1],
                }
            )

        return out

    def get_flgs(self):
        """Get config flags

        Returns:
            list

        """
        return self.get_table(Flg)


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
    edt = TProp(Union[str, Timestamp, datetime], check_datetime)

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

    #: str: Data source
    src = TProp(str)

    def __init__(self, edt, oid, tags=TAG_NONE_NAME, metadata={}, src=''):
        """Constructor

        Args:
            edt (str | datetime | pd.Timestamp): Event datetime (UTC)
            oid (int|iterable of int): Object identifier (snr, pid)
            tags (str|iterable of str, `optional`): Tags. Defaults to ''
            metadata (dict|InfoManagerMetaData, `optional`): Default to {}
            src (str): Default to ''

        """

        # Init
        if isinstance(metadata, dict):
            metadata = InfoManagerMetaData(metadata)

        # Set attributes
        self.edt = edt
        self.oid = oid
        self.tags = tags
        self.metadata = metadata
        self.src = src

    def __copy__(self):
        return self.__class__(self.edt, self.oid.copy(), self.tags.copy())

    @property
    def eid(self):
        """str: Event ID which match 1st corresponding pattern in tags. Defaults to None."""
        try:
            # TODO: the following line triggers a *very* weird pylint Error 1101.
            # I disable it for now ... but someone should really confirm whether this ok or not!
            # fpavogt - 2020.12.09
            out = next(filter(glob_var.eid_pat.match, self.tags))  # pylint: disable=E1101
        except StopIteration:
            out = None
        return out

    @property
    def rid(self):
        """str: Rig ID which match 1st corresponding pattern in tags. Defaults to None."""
        try:
            # TODO: the following line triggers a *very* weird pylint Error 1101.
            # I disable it for now ... but someone should really confirm whether this ok or not!
            # fpavogt - 2020.12.09
            out = next(filter(glob_var.rid_pat.match, self.tags))  # pylint: disable=E1101
        except StopIteration:
            out = None
        return out

    @property
    def mid(self):
        """str: Model identifier"""
        return [arg[TableModel.mid.name] for arg in self.object]

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
                'join_order': [TableModel],
                'where': TableObject.oid.in_(self.oid)
            },
            recurse=True
        )

        # Set output
        out = list(zip(*sorted([
            (
                res[TableObject.oid.name],
                {
                    TableObject.oid.name: res[TableObject.oid.name],
                    TableObject.srn.name: res[TableObject.srn.name],
                    TableObject.pid.name: res[TableObject.pid.name],
                    TableModel.mdl_name.name: res[TableObject.model.name][TableModel.mdl_name.name],
                    TableModel.mdl_desc.name: res[TableObject.model.name][TableModel.mdl_desc.name],
                    TableModel.mid.name: res[TableObject.model.name][TableModel.mid.name]
                }
            )
            for res in qry_res
        ])))[1]

        return out

    def __repr__(self):
        return f'{self}'

    def __str__(self):
        p_printer = pprint.PrettyPrinter()
        return p_printer.pformat(
            (f'edt: {self.edt}', f'oid: {self.oid}',
             f'tags: {self.tags}', f'metadata: {self.metadata}',
             f'src: {self.src}')
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
        """Sort list of InfoManager. Sorting order [edt, srn, tags]

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
        return self.edt, *[str(arg) for arg in self.oid], *self.tags, self.src

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
            - edt (str): Datetime
            - typ_name (str, `optional`): Instrument type (used to create
                instrument entry if missing in DB)
            - srn_field (str): Serial number
            - pid (str): Product identifier
            - tags (list of str): Tags
            - meta_field (dict): Metadata as dict
            - src (str): Data source

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
                model := db_mngr.get_or_none(
                    TableModel,
                    search={
                        'where': TableModel.mdl_name == metadata[TableModel.mdl_name.name]
                    }
                )
            ) is None:
                # TODO
                #  Detail exception
                raise Exception(f"{metadata[TableModel.mdl_name.name]} is missing in DB/InstrumentType")

            # Create instrument entry
            oid = TableObject.create(
                srn=metadata[TableObject.srn.name],
                pid=metadata[TableObject.pid.name],
                model=model
            ).oid

        # Construct InfoManager
        try:
            info = InfoManager(
                edt=metadata[EDT_FLD_NM],
                oid=oid,
                tags=metadata[TAG_FLD_NM],
                metadata=metadata[META_FLD_NM],
                src=metadata[DataSource.src.name]
            )
        except Exception as exc:
            # TODO
            #  Detail exception
            raise Exception(exc)

        return info


class DBCreateError(Exception):
    """Exception class for DB creation error"""


class DBInsertError(Exception):
    """Exception class for DB insert error"""


class DBDirError(Exception):
    """Exception class for DB directory creating error"""
