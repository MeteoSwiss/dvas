"""
This module contains the database interaction functions and classes

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import from python packages
import pprint
import re
from collections import OrderedDict
from math import floor
from threading import Thread
from datetime import datetime
import pytz
from peewee import chunked, DoesNotExist
from peewee import IntegrityError
from playhouse.shortcuts import model_to_dict
from pandas import DataFrame, to_datetime, Timestamp
import sre_yield


# Import from current package
from .model import db
from .model import Instrument, InstrType, EventsInfo
from .model import Parameter, Flag, OrgiDataInfo, Data
from .model import Tag, EventsTags
from ..config.pattern import instr_re, param_re
from ..config.config import instantiate_config_managers
from ..config.config import InstrType as CfgInstrType
from ..config.config import Instrument as CfgInstrument
from ..config.config import Parameter as CfgParameter
from ..config.config import Flag as CfgFlag
from ..config.config import Tag as CfgTag
from ..dvas_helper import DBAccess
from ..dvas_helper import SingleInstanceMetaClass
from ..dvas_helper import TypedProperty
from ..dvas_helper import TimeIt
from .. import dvas_logger as log


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
        """Return config document

        Args:
            key (str): Config manager key

        Returns:
            dict

        """

        array_old = self._cfg_mngr[key].document
        node_gen = self._cfg_mngr[key].NODE_GEN
        array_new = []

        for i, doc in enumerate(array_old):
            sub_array_new = {}
            if node_gen:
                #TODO add max length
                node_gen_val = sre_yield.AllMatches(doc[node_gen])
                sub_array_new.update(
                    {
                        node_gen: list(node_gen_val)
                    }
                )
                for key in filter(lambda x: x != node_gen, doc.keys()):
                    sub_array_new.update(
                        {
                            key: [
                                re.sub(
                                    doc[node_gen],
                                    '{}'.format(doc[key]),
                                    node_gen_val[i].group()
                                )
                                if node_gen_val[i].groups() else
                                doc[key]
                                for i in range(len(node_gen_val))
                            ]
                        }
                    )
            array_new += [
                dict(zip(sub_array_new, i))
                for i in zip(*sub_array_new.values())
            ]

        return array_new


class DatabaseManager(metaclass=SingleInstanceMetaClass):
    """Local DB manager"""

    DB_TABLES = [
        InstrType, Instrument,
        EventsInfo, Parameter,
        Flag, Tag, EventsTags, Data, OrgiDataInfo
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
        db.database.parent.mkdir(mode=777, parents=True, exist_ok=True)

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
    def get_or_none(table, search=None, attr=None):
        """Get from DB

        Args:
            table ():
            search (dict):
            attr (list):

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
                out = qry.get()

                for arg in attr:
                    out = getattr(out, arg)

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
                    log.localdb.insert.error(
                        "instr_id '%s' is missing in DB", event.instr_id
                    )
                    raise

                # Get/Check parameter
                param = Parameter.get_or_none(
                    Parameter.prm_abbr == event.prm_abbr
                )
                if not param:
                    log.localdb.insert.error(
                        "prm_abbr '%s' is missing in DB", event.prm_abbr
                    )
                    raise DBInsertError()

                # Check tag_abbr existence
                try:
                    tag_id_list = []
                    for arg in event.tag_abbr:
                        assert (
                            tag_id := db_mngr.get_or_none(
                                Tag,
                                search={
                                    'join_order': [],
                                    'where': Tag.tag_abbr == arg
                                },
                                attr=['id']
                            )
                        ) is not None
                        tag_id_list.append(tag_id)

                except AssertionError:
                    log.rawcsv_load.error(
                        "tag_abbr '%s' is missing in DB",
                        arg
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
                    EventsTags.insert_many(tag_event_source).execute()

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

    def _get_eventsinfo_id(self, where, prm_abbr):
        """ """

        # Replace field in string
        where = where.replace(
            'event_dt', 'EventsInfo.event_dt'
        )
        where = where.replace(
            'instr_id', 'Instrument.instr_id'
        )
        where = where.replace(
            'tag_abbr', 'Tag.tag_abbr'
        )

        # Replace logical operators
        where = where.replace(' and ', ' & ')
        where = where.replace(' or ', ' | ')
        where = where.replace('not(', '~(')

        # Add paramerter field
        where = f"({where}) & (Parameter.prm_abbr == '{prm_abbr}')"

        # Evaluate
        where_arg = eval(
            where,
            {
                'EventsInfo': EventsInfo,
                'Instrument': Instrument,
                'Tag': Tag,
                'Parameter': Parameter
            }
        )

        # Create query
        qry = (
            EventsInfo.select().distinct().
            join(Instrument).switch(EventsInfo).
            join(Parameter).switch(EventsInfo).
            join(EventsTags).join(Tag).
            where(where_arg)
        )

        with DBAccess(db) as _:
            out = list(qry.iterator())

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
            log.localdb_select.warning(
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
            print_tables = [
                InstrType, Instrument, Flag,
                Parameter, EventsInfo, OrgiDataInfo,
                Tag, EventsTags,
            ]

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


def set_datetime(val):
    """Test and set input argument into datetime.datetime.

    Args:
        val (str | datetime | pd.Timestamp): UTC datetime

    Returns:
        datetime.datetime

    """
    try:
        assert (out := to_datetime(val).to_pydatetime()).tzinfo == pytz.UTC
    except AssertionError:
        raise TypeError('Not UTC or bad datetime format')

    return out


def set_str(val, re_pattern, fullmatch=True):
    """Test and set input argument into str.

    Args:
        val (str): String
        re_pattern (re.Pattern): Compiled regexp pattern
        fullmatch (bool): Apply fullmatch. Default to True.

    Returns:
        str

    """
    try:
        if fullmatch:
            assert re_pattern.fullmatch(val) is not None
        else:
            assert re_pattern.match(val) is not None
    except AssertionError:
        raise TypeError(f"Argument doesn't match {re_pattern.pattern}")

    return val


def set_list_str(val):
    """Test and set input argument into list of str.

    Args:
        val (list of str): Input

    Returns:
        str

    """
    try:
        assert all([isinstance(arg, str) for arg in val]) is True
    except AssertionError:
        raise TypeError(f"Argument is not a list a str")

    return val


class EventManager:
    """Class for create an unique event identifier"""

    #: datetime.datetime: UTC datetime
    event_dt = TypedProperty((str, Timestamp, datetime), set_datetime)
    #: str: Instrument id
    instr_id = TypedProperty(
        str, set_str, args=(instr_re,), kwargs={'fullmatch': True}
    )
    #: str: Parameter abbd
    prm_abbr = TypedProperty(
        str, set_str, args=(param_re,), kwargs={'fullmatch': True}
    )
    #: str: Tag abbr
    tag_abbr = TypedProperty(list, set_list_str)

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
        for key in [
            'event_dt', 'instr_id', 'prm_abbr', 'tag_abbr',
        ]:
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
