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
from collections import OrderedDict
from math import floor
from threading import Thread
from datetime import datetime
from peewee import chunked, DoesNotExist
from peewee import IntegrityError
from peewee import PeeweeException
from playhouse.shortcuts import model_to_dict
import numpy as np
from pandas import DataFrame, Timestamp
from pampy.helpers import Iterable, Union
import sre_yield

# Import from current package
from .model import db
from .model import Instrument, InstrType, EventsInfo
from .model import Parameter, Flag, OrgiDataInfo, Data
from .model import Tag, EventsTags
from ..config.pattern import PARAM_PAT
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
from ..dvas_helper import unzip
from ..dvas_logger import localdb
from ..dvas_environ import glob_var
from ..dvas_environ import path_var as env_path_var


# Define
SQLITE_MAX_VARIABLE_NUMBER = 999
INDEX_NM = Data.index.name
VALUE_NM = Data.value.name
EVENT_DT_NM = EventsInfo.event_dt.name
EVENT_INSTR_NM = EventsInfo.instrument.id.name
EVENT_PARAM_NM = EventsInfo.param.prm_abbr.name

#: int: Database cache size in kB
DB_CACHE_SIZE = 10 * 1024

#: str: Local database file name
DB_FILE_NM = 'local_db.sqlite'


class OneDimArrayConfigLinker:
    """Link to OneDimArrayConfigManager
    config managers."""

    CFG_MNGRS = [CfgParameter, CfgInstrType, CfgInstrument, CfgFlag, CfgTag]

    def __init__(self, cfg_mngrs=None):
        """Constructor

        Args:
            cfg_mngrs (list of OneDimArrayConfigManager): Config managers

        """

        if cfg_mngrs is None:
            cfg_mngrs = self.CFG_MNGRS

        # Set attributes
        self._cfg_mngr = instantiate_config_managers(*cfg_mngrs)

    def get_document(self, key):
        """Return config document.

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
        InstrType, Instrument,
        EventsInfo, Parameter,
        Flag, Tag, EventsTags, Data, OrgiDataInfo
    ]
    DB_TABLES_PRINT = [
        Parameter, InstrType,
        Instrument, Flag,
        Tag
    ]

    def __init__(self, reset_db=False):
        """

        Args:
            reset_db (bool): Force the data base to be reset.

        """

        # Create config linker instance attribute
        self._cfg_linker = OneDimArrayConfigLinker()

        # Create db attribute
        self._db = db

        # Init db
        exists_db = self._init_db(reset_db)

        if exists_db is False:
            self._create_db()

    @property
    def db(self):
        """peewee.SqliteDatabase: Database instance"""
        return self._db

    def _init_db(self, reset):
        """Init db

        Args:
            reset (bool): Reset DB

        Returns:
            bool: DB already exists or not

        """

        # Define
        file_path = env_path_var.local_db_path / DB_FILE_NM

        # Reset
        if reset:
            file_path.unlink(missing_ok=True)

        # Create local DB directory
        if file_path.exists():
            exists = True
        else:
            exists = False
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
        self._db.init(
            file_path,
            pragmas={
                'foreign_keys': True,
                # Set cache to 10MB
                'cache_size': -DB_CACHE_SIZE
            }
        )

        return exists

    #TODO
    # Fix with DB can't be create several times
    def _create_db(self):
        """Method for creating the database.

        Note:
            If database already exists, the old one will be flushed

        """

        with DBAccess(self) as _:

            try:

                # Drop table
                self._drop_tables()

                # Create table
                self._create_tables()

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

    def get_or_none(self, table, search=None, attr=None, get_first=True):
        """Get from DB

        Args:
            table ():
            search (dict):
            attr (list of (list of str)):
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
        return [model_to_dict(qry, recurse=recurse) for qry in query]

    def _drop_tables(self):
        """Drop db tables"""
        self._db.drop_tables(DatabaseManager.DB_TABLES, safe=True)

    def _create_tables(self):
        """Create db tables"""

        self._db.create_tables(DatabaseManager.DB_TABLES, safe=True)

    def _fill_table(self, table, foreign_constraint=None):
        """

        Args:
            table:
            foreign_constraint:

        """

        # Define
        document = self._cfg_linker.get_document(table.__name__)

        # get foreign constraint attribute
        if foreign_constraint:
            for doc in document:
                for arg in foreign_constraint:
                    mdl_cls = arg['class']
                    cmp_res = getattr(mdl_cls, arg['foreign_attr']) == doc[arg['attr']]
                    doc[arg['attr']] = mdl_cls.get_or_none(cmp_res)

        # Insert
        table.insert_many(document).execute()

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

    def add_data(self, index, value, event, prm_abbr, source_info=None):
        """Add profile data to the DB.

        Args:
            index (np.array of int): Data index
            value (np.array of float): Data value
            event (EventManager):
            prm_abbr (str):
            source_info (str, optional): Data source

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
                instr = Instrument.get_or_none(
                    Instrument.sn == event.sn
                )
                if not instr:
                    err_msg = "instr_id '%s' is missing in DB" % (event.sn)
                    localdb.error(err_msg)
                    raise DBInsertError(err_msg)

                # Get/Check parameter
                param = Parameter.get_or_none(
                    Parameter.prm_abbr == prm_abbr
                )
                if not param:
                    err_msg = "prm_abbr '%s' is missing in DB" % (prm_abbr)
                    localdb.error(err_msg)
                    raise DBInsertError(err_msg)

                # Check tag_abbr existence
                try:
                    tag_id_list = [
                        arg[0] for arg in
                        self.get_or_none(
                            Tag,
                            search={
                                'where': Tag.tag_abbr.in_(event.tag_abbr)
                            },
                            attr=[[Tag.id.name]],
                            get_first=False
                        )
                    ]
                    assert len(tag_id_list) == len(event.tag_abbr)

                except AssertionError:
                    raise DBInsertError(
                        f"Many tag_abbr in {event.tag_abbr} are missing in DB",
                    )

                # Create original data information
                orig_data_info, _ = OrgiDataInfo.get_or_create(
                    source=source_info)

                # Create event info
                event_info, created = EventsInfo.get_or_create(
                    event_dt=event.event_dt, instrument=instr,
                    param=param, orig_data_info=orig_data_info
                )

                # Insert data
                if created:

                    # Link event to tag
                    tag_event_source = [
                        {
                            EventsTags.tag.name: tag_id,
                            EventsTags.events_info.name: event_info
                        } for tag_id in tag_id_list
                    ]
                    EventsTags.insert_many(tag_event_source).execute()  # noqa pylint: disable=E1120

                    # Create batch index
                    fields = [Data.index, Data.value, Data.event_info]

                    # Create batch data
                    batch_data = zip(
                        index,
                        value,
                        [event_info] * len(value)
                    )

                    # Calculate max batch size
                    n_max = floor(SQLITE_MAX_VARIABLE_NUMBER / len(fields))

                    # Insert to db
                    for batch in chunked(batch_data, n_max):
                        Data.insert_many(batch, fields=fields).execute()  # noqa pylint: disable=E1120

        except DBInsertError as exc:
            raise DBInsertError(exc)

    @staticmethod
    def _get_eventsinfo_id(where_arg, prm_abbr, filter_empty):
        """Get events id"""

        try:
            out = list(SearchEventExpr.eval(where_arg, prm_abbr, filter_empty))

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
            prm_abbr:
            filter_empty:

        Returns:

        """
        # Get event_info id
        eventsinfo_id_list = self._get_eventsinfo_id(where, prm_abbr, filter_empty)

        if not eventsinfo_id_list:
            localdb.warning(
                "Empty search '%s' for '%s", where, prm_abbr
            )

        # Query data
        qryer = []

        for eventsinfo_id in eventsinfo_id_list:
            qryer.append(Queryer(self, eventsinfo_id))

        with DBAccess(self) as _:
            for qry in qryer:
                qry.start()
                qry.join()

            # Group data
            out = []
            for i, qry in enumerate(qryer):
                tag_abbr = [
                    arg.tag_abbr for arg in
                    Tag.select().distinct().
                    join(EventsTags).join(EventsInfo).
                    where(EventsInfo.id == eventsinfo_id_list[i].id).
                    iterator()
                ]
                out.append(
                    {
                        'event': EventManager(
                            event_dt=eventsinfo_id_list[i].event_dt,
                            sn=eventsinfo_id_list[i].instrument.sn,
                            tag_abbr=tag_abbr,
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
        """Get config flags"""
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

    def __init__(self, db_mngr, eventsinfo_id):

        # Call super constructor
        super().__init__()

        # Init attributes
        self._eventsinfo_id = eventsinfo_id
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
                where(Data.event_info == self._eventsinfo_id)
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


class EventManager:
    """Class for create an unique event identifier"""

    #: datetime.datetime: UTC datetime
    event_dt = TProp(Union[str, Timestamp, datetime], check_datetime)

    #: str: Instrument id
    sn = TProp(str, lambda *x: x[0])

    #: str: Tag abbr
    tag_abbr = TProp(
        Union[None, Iterable[str]],
        setter_fct=lambda x: set(x) if x else set(),
        getter_fct=lambda x: sorted(x)
    )

    def __init__(self, event_dt, sn, tag_abbr=None):
        """Constructor

        Args:
            event_dt (str | datetime | pd.Timestamp): UTC datetime
            sn (str): Instrument serial number
            tag_abbr (`optional`, iterable of str): Tag abbr iterable

        """

        # Set attributes
        self.event_dt = event_dt
        self.sn = sn
        self.tag_abbr = tag_abbr

    def __repr__(self):
        p_printer = pprint.PrettyPrinter()
        return p_printer.pformat(
            (f'dt: {self.event_dt}', f'sn: {self.sn}', f'tag: {self.tag_abbr}')
        )

    def __hash__(self):
        return hash(self.sort_attr)

    def add_tag(self, val):
        """Add a tag abbr

        Args:
            val (str): New tag abbr

        """

        #TODO
        # consider to use the observer pattern
        # Modify tag 'raw' -> 'derived'
        #self.rm_tag(TAG_RAW_VAL)
        #self.add_tag(TAG_DERIVED_VAL)

        # Add new tag
        self.tag_abbr = self.tag_abbr + [val,]

    def rm_tag(self, val):
        """Remove a tag abbr

        Args:
            val (str): Tag abbr to remove

        """
        if self.tag_abbr.intersection({val}):
            self.tag_abbr.remove(val)

    @staticmethod
    def sort(event_list):
        """Sort list of event manager. Sorting order [event_dt, instr_id]

        Args:
            event_list (list of EventManager): List to sort

        Returns:
            list: Sorted event manager list
            tuple: Original list index

        """

        # Zip index
        val = list(zip(event_list, range(len(event_list))))

        # Sort
        val.sort()

        # Unzip index
        out = list(unzip(val))

        return list(out[0]), out[1]

    @property
    def sort_attr(self):
        """ list of EventManger attributes: Attributes sort order"""
        return tuple((self.event_dt, self.sn, *self.tag_abbr))

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


class SearchEventExpr(metaclass=ABCMeta):
    """Abstract search event expression interpreter class.

    .. uml::

        @startuml
        footer Interpreter design pattern

        class SearchEventExpr {
            {abstract} interpret()
            {static} eval()
        }

        class LogicalSearchEventExpr {
            _expression: List
            interpret()
            {abstract} fct(*arg)
        }

        SearchEventExpr <|-- LogicalSearchEventExpr : extends
        LogicalSearchEventExpr o--> SearchEventExpr

        class TerminalSearchEventExpr {
            interpret()
            {abstract} get_filter()
        }

        SearchEventExpr <|-- TerminalSearchEventExpr : extends

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
            List of EventsInfo

        Search expression grammar:
            - all(): Select all
            - [datetime ; dt]('<ISO datetime>', ['=='(default) ; '>=' ; '>' ; '<=' ; '<' ; '!=']): Select by datetime
            - [serialnumber ; sn]('<Serial number>'): Select by serial number
            - tag(['<Tag>' ; ('<Tag 1>', ...,'<Tag n>')]): Select by tag
            - and_(<expr 1>, ..., <expr n>): Intersection
            - or_(<expr 1>, ..., <expr n>): Union
            - not_(<expr>): Negation, correspond to all() without <expr>
        """

        # Define
        str_expr_dict = {
            'all': AllExpr,
            'datetime': DatetimeExpr, 'dt': DatetimeExpr,
            'serialnumber': SerialNumberExpr, 'sn': SerialNumberExpr,
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
            qry = EventsInfo.select().where(EventsInfo.id.in_(expr_res))
            out = [arg for arg in qry.iterator()]

            return out


class LogicalSearchEventExpr(SearchEventExpr):
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


class AndExpr(LogicalSearchEventExpr):
    """And operation"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.and_(a, b)


class OrExpr(LogicalSearchEventExpr):
    """Or operation"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.or_(a, b)


class NotExpr(LogicalSearchEventExpr):
    """Not operation"""

    def __init__(self, arg):
        self._expression = [AllExpr(), arg]

    def fct(self, a, b):
        """Implement fct method"""
        return operator.sub(a, b)


class TerminalSearchEventExpr(SearchEventExpr):
    """
    Implement an interpret operation associated with terminal symbols in
    the grammar.
    """

    QRY_BASE = (
        EventsInfo
        .select().distinct()
        .join(Instrument).switch(EventsInfo)
        .join(Parameter).switch(EventsInfo)
        .join(EventsTags).join(Tag)
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


class AllExpr(TerminalSearchEventExpr):
    """All filter"""

    def __init__(self):
        pass

    def get_filter(self):
        """Implement get_filter method"""
        return


class DatetimeExpr(TerminalSearchEventExpr):
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
        return self._op(EventsInfo.event_dt, self.expression)


class SerialNumberExpr(TerminalSearchEventExpr):
    """Serial number filter"""

    expression = TProp(str, lambda *x: x[0])

    def get_filter(self):
        """Implement get_filter method"""
        return Instrument.sn == self.expression


class TagExpr(TerminalSearchEventExpr):
    """Tag filter"""

    expression = TProp(
        Union[str, Iterable[str]], lambda x: set([x]) if isinstance(x, str) else set(x)
    )

    def get_filter(self):
        """Implement get_filter method"""
        return Tag.tag_abbr.in_(self.expression)


class ParameterExpr(TerminalSearchEventExpr):
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
