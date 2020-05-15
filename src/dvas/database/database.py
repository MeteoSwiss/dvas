"""
This module contains the database interaction functions and classes

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import from python packages
import pprint
from math import floor
from threading import Thread
import pytz
from peewee import chunked, DoesNotExist
from peewee import IntegrityError
from playhouse.shortcuts import model_to_dict
import pandas as pd

# Import from current package
from .model import db
from .model import Instrument, InstrType, EventsInstrumentsParameters
from .model import Parameter, Flag, OrgiDataInfo, Data
from ..config.pattern import PARAM_KEY, INSTR_KEY, batch_re
from ..config.pattern import INSTR_TYPE_KEY, instr_re, param_re
from ..config.pattern import FLAG_KEY
from ..config.config import instantiate_config_managers
from ..config.config import InstrType as CfgInstrType
from ..config.config import Instrument as CfgInstrument
from ..config.config import Parameter as CfgParameter
from ..config.config import Flag as CfgFlag
from ..dvas_helper import DBAccess, SingleInstanceMetaClass
from ..dvas_helper import TimeIt


# Define
SQLITE_MAX_VARIABLE_NUMBER = 999

INDEX_NM = Data.index.name
VALUE_NM = Data.value.name
EVENT_DT_NM = EventsInstrumentsParameters.event_dt.name
EVENT_INSTR_NM = EventsInstrumentsParameters.instrument.id.name
EVENT_PARAM_NM = EventsInstrumentsParameters.param.prm_abbr.name
EVENT_ID_NM = EventsInstrumentsParameters.event_id
EVENT_BATCH_NM = EventsInstrumentsParameters.batch_id.name


class DatabaseManager(metaclass=SingleInstanceMetaClass):
    """Local DB manager"""

    def create_db(self):
        """

        Args:
            cfg_linker:

        Returns:

        """

        # Create config linker instance
        cfg_linker = ConfigLinker()

        # Create local DB directory
        db.database.parent.mkdir(mode=777, parents=True, exist_ok=True)

        with DBAccess(db) as _:

            try:

                # Drop table
                self._drop_tables()

                # Create table
                self._create_tables()

                # Fill parameters
                self._fill_table(Parameter, cfg_linker.get_parameters())

                # Fill instr_types
                self._fill_table(InstrType, cfg_linker.get_instr_types())

                # File instruments
                self._fill_table(
                    Instrument, cfg_linker.get_instruments(),
                    foreign_constraint=[
                        {'attr': Instrument.instr_type.name,
                         'class': InstrType,
                         'foreign_attr': InstrType.type_name.name
                        },
                    ]
                )

                # Fill Flag table
                self._fill_table(Flag, cfg_linker.get_flags())

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
        db.drop_tables(
            [
                InstrType, Instrument,
                EventsInstrumentsParameters, Parameter,
                Flag, Data, OrgiDataInfo
            ],
            safe=True
        )

    @staticmethod
    def _create_tables():
        """Create db tables"""

        db.create_tables(
            [
                InstrType, Instrument,
                EventsInstrumentsParameters, Parameter,
                Flag, Data, OrgiDataInfo
            ],
            safe=True
        )

    @staticmethod
    def _fill_table(table, document, foreign_constraint=None):
        """

        Args:
            table:
            document:
            foreign_constraint:

        """

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

        with DBAccess(db) as _:

            # Get/Check instrument
            instr = Instrument.get_or_none(
                Instrument.instr_id == event.instr_id
            )
            if not instr:
                raise Exception(f'{event.instr_id} not found')

            # Get/Check parameter
            param = Parameter.get_or_none(
                Parameter.prm_abbr == event.prm_abbr
            )
            if not param:
                raise Exception(f'{event.prm_abbr} not found')

            # Create original data information
            orig_data_info, _ = OrgiDataInfo.get_or_create(
                source=source_info)

            # Create event-instr-param
            event_instr_param, created = EventsInstrumentsParameters.get_or_create(
                event_dt=event.event_dt, instrument=instr,
                param=param, event_id=event.event_id,
                batch_id=event.batch_id, day_event=event.day_event,
                orig_data_info=orig_data_info
            )

            # Insert data
            if created:

                # Format series as list of tuples
                n_data = len(data)
                data = pd.DataFrame(data, columns=[VALUE_NM])
                data.index.name = INDEX_NM
                data.reset_index(inplace=True)
                data = data.to_dict(orient='list')
                data = list(
                    zip(
                        data[INDEX_NM],
                        data[VALUE_NM],
                        [event_instr_param]*n_data
                    )
                )

                # Create batch index
                fields = [Data.index, Data.value, Data.event_instr_param]
                n_max = floor(SQLITE_MAX_VARIABLE_NUMBER / len(fields))

                # Insert to db
                for batch in chunked(data, n_max):
                    Data.insert_many(batch, fields=fields).execute()  # noqa pylint: disable=E1120

    @staticmethod
    def _get_eventinstrprm(where, prm_abbr):
        """ """

        # Replace field in string
        where = where.replace(
            'event_dt', 'EventsInstrumentsParameters.event_dt')
        where = where.replace(
            'instr_id', 'Instrument.instr_id')
        where = where.replace(
            'batch_id', 'EventsInstrumentsParameters.batch_id')
        where = where.replace(
            'day_event', 'EventsInstrumentsParameters.day_event')
        where = where.replace(
            'event_id', 'EventsInstrumentsParameters.event_id')

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
                'EventsInstrumentsParameters': EventsInstrumentsParameters,
                'Instrument': Instrument,
                'Parameter': Parameter
            }
        )

        # Create query
        qry = (
            EventsInstrumentsParameters.
            select().
            join(Instrument).
            switch(EventsInstrumentsParameters).
            join(Parameter).
            switch(EventsInstrumentsParameters).
            where(where_arg)
        )

        with DBAccess(db) as _:
            out = list(qry.iterator())

        return out

    @TimeIt()
    def get_data(self, where, prm_abbr):
        """

        Args:
            where:
            prm_abbr:

        Returns:

        """
        # Get event intrument parameter id
        eventinstrprm_list = self._get_eventinstrprm(where, prm_abbr)

        if not eventinstrprm_list:
            raise DoesNotExist()

        # Query data
        qryer = []

        for eventinstrprm in eventinstrprm_list:
            qryer.append(Queryer(eventinstrprm))

        with DBAccess(db) as _:
            for qry in qryer:
                qry.start()
                qry.join()

        # Group data
        out = []
        for qry in qryer:
            out.append(qry.res)

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
                Parameter, EventsInstrumentsParameters, OrgiDataInfo
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

    def __init__(self, eventinstrprm):
        """Constructor"""
        super().__init__()
        self.eventinstrprm = eventinstrprm
        self.res = None
        self.exc = None

    def run(self):
        """Run method"""

        try:

            qry = (
                Data.
                select(Data.index, Data.value).
                where(Data.event_instr_param == self.eventinstrprm)
            )

            with DBAccess(db, close_by_exit=False):
                data = list(qry.tuples().iterator())

            # Convert to data frame
            data = pd.DataFrame(
                data,
                columns=[INDEX_NM, VALUE_NM]
            )
            self.res = {
                'event': EventManager(
                    event_dt=self.eventinstrprm.event_dt,
                    instr_id=self.eventinstrprm.instrument.instr_id,
                    prm_abbr=self.eventinstrprm.param.prm_abbr,
                    batch_id=self.eventinstrprm.batch_id,
                    day_event=self.eventinstrprm.day_event,
                    event_id=self.eventinstrprm.event_id,

                ),
                'data': data
            }

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


class ConfigLinker:
    """Link with YAML config files"""

    def __init__(self):
        """Constructor"""
        self._cfg_mngr = instantiate_config_managers(
            CfgParameter, CfgInstrType, CfgInstrument, CfgFlag
        )

    @property
    def cfg_mngr(self):
        """dict: Config managers"""
        return self._cfg_mngr

    def get_parameters(self):
        """dict: Config parameters values"""
        return self.cfg_mngr[PARAM_KEY].document

    def get_instr_types(self):
        """dict: Config instrument type values"""
        return self.cfg_mngr[INSTR_TYPE_KEY].document

    def get_instruments(self):
        """dict: Config instrument values"""
        return self.cfg_mngr[INSTR_KEY].document

    def get_flags(self):
        """dict: Config flag values"""
        return self.cfg_mngr[FLAG_KEY].document


class EventManager:
    """Class for create an unique event identifier"""

    def __init__(self, event_dt, instr_id, prm_abbr, batch_id, day_event, event_id=''):
        """Constructor

        Args:
            event_dt (str | datetime | pd.Timestamp): UTC datetime
            instr_id (str):
            prm_abbr (str):
            batch_id (str):
            day_event (bool):
            event_id (str, optional): Default to ''

        """

        # Set attributes
        # --------------

        # Set datetime
        if isinstance(event_dt, pd.Timestamp):
            value = event_dt.to_pydatetime()
        elif isinstance(event_dt, str):
            value = pd.to_datetime(event_dt).to_pydatetime()
        else:
            raise AttributeError('Bad type')

        # Check time zone (UTC)
        assert value.tzinfo == pytz.UTC, ('Not UTC datetime')
        self._event_dt = event_dt

        # Set instrument id
        if instr_re.match(instr_id) is None:
            raise AttributeError('Bad pattern')
        self._instr_id = instr_id

        # Set instrument id
        if param_re.match(prm_abbr) is None:
            raise AttributeError('Bad pattern')
        self._prm_abbr = prm_abbr

        self._event_id = event_id

        # Set batch id
        if batch_re.match(batch_id) is None:
            raise AttributeError('Bad pattern')
        self._batch_id = batch_id

        # Set day event
        self._day_event = day_event

    @property
    def event_dt(self):
        """datetime: UTC event datetime"""
        return self._event_dt

    @property
    def instr_id(self):
        """str: Instrument id"""
        return self._instr_id

    @property
    def prm_abbr(self):
        """str: Parameter abbreviation"""
        return self._prm_abbr

    @property
    def batch_id(self):
        """str: Batch id"""
        return self._batch_id

    @property
    def day_event(self):
        """bool: Event is a day event or not"""
        return self._day_event

    @property
    def event_id(self):
        """str: Event id"""
        return self._event_id

    def __repr__(self):
        p_printer = pprint.PrettyPrinter()
        return p_printer.pformat(self.as_dict())

    def as_dict(self):
        """Convert EventManager to dict"""
        out = {
            key: self.__getattribute__(key)
            for key in [
                'event_dt', 'instr_id', 'prm_abbr',
                'event_id', 'batch_id', 'day_event',
            ]
        }
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
        return [self.event_dt, self.instr_id]

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
