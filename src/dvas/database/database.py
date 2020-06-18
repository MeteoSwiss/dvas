"""
This module contains the database interaction functions and classes

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import from python packages
import pprint
import re
from itertools import chain, zip_longest
from collections import OrderedDict
from math import floor
from threading import Thread
from datetime import datetime
from peewee import chunked, DoesNotExist
from peewee import IntegrityError
from playhouse.shortcuts import model_to_dict
from pandas import DataFrame, to_datetime, Timestamp
from pampy.helpers import Iterable, Union
import sre_yield
import oschmod

# Import from current package
from .model import db
from .model import Instrument, InstrType, EventsInfo
from .model import Parameter, Flag, OrgiDataInfo, Data
from .model import Tag, EventsTags
from ..config.pattern import INSTR_PAT, PARAM_PAT
from ..config.config import instantiate_config_managers
from ..config.config import InstrType as CfgInstrType
from ..config.config import Instrument as CfgInstrument
from ..config.config import Parameter as CfgParameter
from ..config.config import Flag as CfgFlag
from ..config.config import Tag as CfgTag
from ..dvas_helper import DBAccess
from ..dvas_helper import SingleInstanceMetaClass
from ..dvas_helper import TypedProperty as TProp
from ..dvas_helper import TimeIt
from ..dvas_helper import get_by_path, check_datetime
from ..dvas_logger import localdb, rawcsv
from ..dvas_environ import glob_var


# Define
SQLITE_MAX_VARIABLE_NUMBER = 999
INDEX_NM = Data.index.name
VALUE_NM = Data.value.name
EVENT_DT_NM = EventsInfo.event_dt.name
EVENT_INSTR_NM = EventsInfo.instrument.id.name
EVENT_PARAM_NM = EventsInfo.param.prm_abbr.name


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
    """Local DB manager"""

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

    def __init__(self):
        """Constructor"""

        # Create config linker instance
        self._cfg_linker = OneDimArrayConfigLinker()

    def create_db(self):
        """Method for creating the database.

        Note:
            If database already exists, the old one will be flushed

        """

        # Create local DB directory
        try:
            db.database.parent.mkdir(parents=True, exist_ok=True)
            oschmod.set_mode(db.database.parent.as_posix(), 'u+rw')
        except (OSError,) as exc:
            raise DBDirError(
                f"Error in creating '{db.database.parent}' ({exc})"
            )

        with DBAccess(db) as _:

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

            except IntegrityError:
                pass

    @staticmethod
    def get_or_none(table, search=None, attr=None, get_first=True):
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
            with DBAccess(db) as _:
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

    @staticmethod
    def _drop_tables():
        """Drop db tables"""
        db.drop_tables(DatabaseManager.DB_TABLES, safe=True)

    @staticmethod
    def _create_tables():
        """Create db tables"""

        db.create_tables(DatabaseManager.DB_TABLES, safe=True)

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

        with DBAccess(db) as _:
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

    @staticmethod
    def add_data(data, event, source_info=None):
        """Add data

        Args:
            data (pd.Series):
            event (EventManager):
            source_info (str, optional): Data source

        """

        try:
            with DBAccess(db) as _:

                # Get/Check instrument
                instr = Instrument.get_or_none(
                    Instrument.instr_id == event.instr_id
                )
                if not instr:
                    localdb.insert.error(
                        "instr_id '%s' is missing in DB", event.instr_id
                    )
                    raise DBInsertError()

                # Get/Check parameter
                param = Parameter.get_or_none(
                    Parameter.prm_abbr == event.prm_abbr
                )
                if not param:
                    localdb.insert.error(
                        "prm_abbr '%s' is missing in DB", event.prm_abbr
                    )
                    raise DBInsertError()

                # Check tag_abbr existence
                try:
                    tag_id_list = [
                        arg[0] for arg in
                        db_mngr.get_or_none(
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
                    rawcsv.error(
                        "Many tag_abbr in '%s' are missing in DB",
                        event.tag_abbr
                    )
                    raise DBInsertError()

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

                    # Format series as list of tuples
                    n_data = len(data)
                    data = DataFrame(data, columns=[VALUE_NM])
                    data.index.name = INDEX_NM
                    data.reset_index(inplace=True)
                    data = data.to_dict(orient='list')
                    data = list(
                        zip(
                            data[INDEX_NM],
                            data[VALUE_NM],
                            [event_info]*n_data
                        )
                    )

                    # Create batch index
                    fields = [Data.index, Data.value, Data.event_info]
                    n_max = floor(SQLITE_MAX_VARIABLE_NUMBER / len(fields))

                    # Insert to db
                    for batch in chunked(data, n_max):
                        Data.insert_many(batch, fields=fields).execute()  # noqa pylint: disable=E1120

        except DBInsertError:
            pass

    @staticmethod
    def _get_eventsinfo_id(where_arg, prm_abbr):
        """Get events info

        Search syntax:
            - Logical and: and, &
            - Logical or: or, |
            - Operators: ==, >=, <=, <, >, !=
            - Event datetime field: #e, #event, #event_dt
            - Instrument field: #i, #instr, #instr_id
            - Tag field: #t, #instr, #instr_id
            - Datetime: %'any valid pandas.datetime'%

        """

        # Substitute and and or
        where_arg = re.sub(r'[\s\t]+(and)[\s\t]+', '&', where_arg)
        where_arg = re.sub(r'[\s\t]+(or)[\s\t]+', '|', where_arg)

        # Substitute spaces
        where_arg = re.sub(r'[\s\t\n\r]+', '', where_arg)

        # Split and find kernel logical conditions
        where_split = re.split(r"[\#\w\=\>\<\!\'\"\[\]\,\-\:\%]+", where_arg)
        where_find = re.findall(r"([\#\w\=\>\<\!\'\"\[\]\,\-\:\%]+)", where_arg)

        # Create base request
        qry_base = (
            EventsInfo
            .select().distinct()
            .join(Instrument).switch(EventsInfo)
            .join(Parameter).switch(EventsInfo)
            .join(EventsTags).join(Tag)
        )

        try:
            with DBAccess(db) as _:
                # Search for kernel logical condition
                search_res = []
                for where in where_find:
                    # Replace field in string
                    where = re.sub(
                        r'\#((e(vent(_dt)?)?)|(dt))', 'EventsInfo.event_dt', where
                    )
                    where = re.sub(
                        r'\#(i(nstr(_id)?)?)', 'Instrument.instr_id', where
                    )
                    where = re.sub(
                        r'\#(t(ag(_abbr)?)?)', 'Tag.tag_abbr', where
                    )

                    # Replace datetime
                    where = re.sub(
                        r'\%([\dTZ\:\-]+)\%', r"to_datetime('\1').to_pydatetime()", where
                    )

                    # Create query
                    qry_tmp = qry_base.where(
                        eval(
                            where,
                            {
                                'EventsInfo': EventsInfo,
                                'Instrument': Instrument,
                                'Tag': Tag,
                                'to_datetime': to_datetime
                            }
                        ) &
                        (Parameter.prm_abbr == prm_abbr)
                    )

                    # Execute query
                    search_res.append(
                        str(
                            set(arg.id for arg in qry_tmp.iterator())
                        )
                    )

                # Eval set logical expression
                out = list(eval(''.join(
                    [arg for arg in chain(
                        *zip_longest(

                            # Split formula
                            where_split,

                            # Find formula and substitute
                            search_res
                        )
                    ) if arg]
                )))

                # Convert id as table element
                qry = EventsInfo.select().where(EventsInfo.id.in_(out))
                out = [arg for arg in qry.iterator()]

        except Exception as exc:
            print(where)
            print(where_find)
            print(exc)
            out = []

        return out

    @TimeIt()
    def get_data(self, where, prm_abbr):
        """Get data from DB

        Args:
            where:
            prm_abbr:

        Returns:

        """
        # Get event_info id
        eventsinfo_id_list = self._get_eventsinfo_id(where, prm_abbr)

        if not eventsinfo_id_list:
            localdb.warning(
                "Empty search '%s' for '%s", where, prm_abbr
            )

        # Query data
        qryer = []

        for eventsinfo_id in eventsinfo_id_list:
            qryer.append(Queryer(eventsinfo_id))

        with DBAccess(db) as _:
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
                            instr_id=eventsinfo_id_list[i].instrument.instr_id,
                            prm_abbr=eventsinfo_id_list[i].param.prm_abbr,
                            tag_abbr=tag_abbr,
                        ),
                        'data': qry.res
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


class Queryer(Thread):
    """Data queryer"""

    def __init__(self, eventsinfo_id):
        """Constructor"""
        super().__init__()
        self.eventsinfo_id = eventsinfo_id
        self.res = None
        self.exc = None

    def run(self):
        """Run method"""

        try:
            qry = (
                Data.
                select(Data.index, Data.value).
                where(Data.event_info == self.eventsinfo_id)
            )

            with DBAccess(db, close_by_exit=False):
                data = list(qry.tuples().iterator())

            # Convert to data frame
            data = DataFrame(
                data,
                columns=[INDEX_NM, VALUE_NM]
            )
            self.res = data

        except Exception as ex:
            self.exc = Exception(ex)

    def join(self, timeout=None):
        """Overwrite join method"""
        super().join(timeout=None)

        # Test exception attribute
        if self.exc:
            raise self.exc


#: DatabaseManager: Local SQLite database manager
db_mngr = DatabaseManager()


class EventManager:
    """Class for create an unique event identifier"""

    #: datetime.datetime: UTC datetime
    event_dt = TProp(Union[str, Timestamp, datetime], check_datetime)
    #: str: Instrument id
    instr_id = TProp(re.compile(rf'^({INSTR_PAT})$'), lambda *x: x[0])
    #: str: Parameter abbd
    prm_abbr = TProp(re.compile(rf'^({PARAM_PAT})$'), lambda *x: x[0])
    #: str: Tag abbr
    tag_abbr = TProp(Iterable[str], lambda x: list(x))  #TODO use set()

    def __init__(self, event_dt, instr_id, prm_abbr, tag_abbr):
        """Constructor

        Args:
            event_dt (str | datetime | pd.Timestamp): UTC datetime
            instr_id (str):
            prm_abbr (str):
            tag_abbr (list of str):

        """

        # Set attributes
        self.event_dt = event_dt
        self.instr_id = instr_id
        self.prm_abbr = prm_abbr
        self.tag_abbr = tag_abbr

    def __repr__(self):
        p_printer = pprint.PrettyPrinter()
        return p_printer.pformat(self.as_dict())

    def as_dict(self):
        """Convert EventManager to dict"""
        out = OrderedDict()
        keys_nm = ['event_dt', 'instr_id', 'prm_abbr', 'tag_abbr',]
        for key in keys_nm:
            out.update({key: self.__getattribute__(key)})
        return out

    @staticmethod
    def sort(event_list):
        """Sort list of event manager. Sorting order [event_dt, instr_id]

        Args:
            event_list (list of EventManager):

        Returns:

        """
        event_list.sort()

    @property
    def sort_attr(self):
        """ list of EventManger attributes: Attributes sort order"""
        return [self.event_dt, self.instr_id, self.prm_abbr]

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


class DBInsertError(Exception):
    """Exception class for DB insert error"""

class DBDirError(Exception):
    """Exception class for DB directory creating error"""


class ConfigGenMaxLenError(Exception):
    """Exception class for max length config generator error"""
