"""

"""

# Import from python packages
import pprint
from datetime import datetime
import pytz
from dataclasses import dataclass, field, asdict
from math import floor
from functools import wraps
import pandas as pd
from threading import Thread
from mdtpyhelper.misc import timer
import peewee
from peewee import chunked, DoesNotExist
from playhouse.shortcuts import model_to_dict

# Import from current package
from .model import db, DATABASE_FILE_PATH
from .model import Instrument, InstrType, EventsInstrumentsParameters
from .model import Parameter, Flag, OrgiDataInfo, Data
from ..config.pattern import PARAM_KEY, INSTR_KEY
from ..config.pattern import INSTR_TYPE_KEY
from ..config.pattern import FLAG_KEY
from ..config.config import instantiate_config_managers
from ..config.config import InstrType as CfgInstrType
from ..config.config import Instrument as CfgInstrument
from ..config.config import Parameter as CfgParameter
from ..config.config import Flag as CfgFlag

SQLITE_MAX_VARIABLE_NUMBER = 999

INDEX_NM = Data.index.name
VALUE_NM = Data.value.name
EVENT_DT_NM = EventsInstrumentsParameters.event_dt.name
EVENT_INSTR_NM = EventsInstrumentsParameters.instrument.id.name
EVENT_PARAM_NM = EventsInstrumentsParameters.param.prm_abbr.name
EVENT_ID_NM = EventsInstrumentsParameters.event_id
EVENT_BATCH_NM = EventsInstrumentsParameters.batch_id.name


# TODO
#  Remove verbose?
#  Add log?
def db_context(verbose=False):
    """ """

    def decorator(func):
        """

        Args:
            func:

        Returns:

        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """

            Args:
                *args:
                **kwargs:

            Returns:

            """
            try:
                with db.connection_context():
                    return func(*args, **kwargs)
            except Exception as e:
                raise Exception(e)

        return wrapper
    return decorator


class DatabaseManager:
    _INSTANCES = 0

    def __init__(self):

        if DatabaseManager._INSTANCES > 0:
            errmsg = f'More than one instance of class {self.__class__.__name__} has been created'
            raise Exception(errmsg)
        else:
            DatabaseManager._INSTANCES += 1

    def create_db(self):
        """

        Args:
            cfg_linker:

        Returns:

        """

        # Create config linker instance
        cfg_linker = ConfigLinker()

        # Erase db
        if DATABASE_FILE_PATH.exists():
            DATABASE_FILE_PATH.unlink()

        # Create db
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

    @db_context()
    def get_or_none(self, table, search, attr):
        """ """

        qry = table.select()
        if search:

            if 'join_order' not in search.keys():
                search['join_order'] = []

            for jointbl in search['join_order']:
                qry = qry.join(jointbl)
                qry = qry.switch(table)
            qry = qry.where(search['where'])

        try:
            out = qry.get()

            for arg in attr:
                out = getattr(out, arg)

        except DoesNotExist:
            out = None

        return out

    @staticmethod
    def _model_to_dict(query, recurse=False):
        return [model_to_dict(qry, recurse=recurse) for qry in query]

    @db_context()
    def _create_tables(self):
        db.create_tables(
            [InstrType, Instrument,
             EventsInstrumentsParameters, Parameter,
             Flag, Data, OrgiDataInfo],
            safe=True
        )

    @db_context()
    def _fill_table(self, table, document, foreign_constraint=None):
        """

        Args:
            table:
            document:
            foreign_constraint:

        Returns:

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

    @db_context()
    def _get_table(self, table, search=None, recurse=False):
        """

        Args:
            table:
            search: [join_order (optional), where]

        Returns:

        """

        qry = table.select()
        if search:

            if 'join_order' not in search.keys():
                search['join_order'] = []

            for jointbl in search['join_order']:
                qry = qry.join(jointbl)
                qry = qry.switch(table)
            qry = qry.where(search['where'])

        return self._model_to_dict(qry, recurse=recurse)

    @db_context()
    def add_data(self, data, event, source_info=None):
        """

        Args:
            data (pd.Series):
            event (EventManager):

        """

        # Get/Check instrument
        instr = Instrument.get_or_none(Instrument.instr_id == event.instr_id)
        if not instr:
            raise Exception(f'{event.instr_id} not found')

        # Get/Check parameter
        param = Parameter.get_or_none(Parameter.prm_abbr == event.prm_abbr)
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
            data = list(zip(
                data[INDEX_NM],
                data[VALUE_NM],
                [event_instr_param]*n_data)
            )

            # Create batch index
            fields = [Data.index, Data.value, Data.event_instr_param]
            N_MAX = floor(SQLITE_MAX_VARIABLE_NUMBER / len(fields))

            # Insert to db
            for batch in chunked(data, N_MAX):
                Data.insert_many(batch, fields=fields).execute()

    @db_context()
    def _get_eventinstrprm(self, where, prm_abbr):
        """ """

        # Replace field in string
        where = where.replace(
            'event_dt', 'EventsInstrumentsParameters.event_dt')
        where = where.replace(
            'instr_id', 'Instrument.id')
        where = where.replace(
            'event_id', 'EventsInstrumentsParameters.event_id')
        where = where.replace(
            'batch_id', 'EventsInstrumentsParameters.batch_id')
        where = where.replace(
            'day_event', 'EventsInstrumentsParameters.day_event')
        where += f" and (Parameter.prm_abbr == '{prm_abbr}')"

        where_arg = eval(where)

        # Create query
        qry = (
            EventsInstrumentsParameters.
            select().
            join(Instrument).
            switch(EventsInstrumentsParameters).
            join(Parameter).
            where(where_arg)
        )

        return [arg for arg in qry.iterator()]

    def get_data(self, where, prm_abbr):

        eventinstrprm_list = self._get_eventinstrprm(where, prm_abbr)

        if not eventinstrprm_list:
            raise peewee.DoesNotExist()

        class Queryer(Thread):

            def __init__(self, eventinstrprm):
                super().__init__()
                self.eventinstrprm = eventinstrprm
                self.res = None
                self.exc = None

            @db_context()
            def run(self):
                try:
                    qry = (
                        Data.
                        select(Data.index, Data.value).
                        where(Data.event_instr_param == self.eventinstrprm)
                    )

                    # Convert to data frame
                    data = [arg for arg in qry.tuples().iterator()]
                    data = pd.DataFrame(
                        data,
                        columns=[INDEX_NM, VALUE_NM]
                    )
                    self.res = {
                        'event': EventManager(
                            event_dt=self.eventinstrprm.event_dt,
                            instr_id=self.eventinstrprm.instrument.id,
                            prm_abbr=self.eventinstrprm.param.prm_abbr,
                            batch_id=self.eventinstrprm.batch_id,
                            day_event=self.eventinstrprm.day_event,
                            event_id=self.eventinstrprm.event_id,

                        ),
                        'data': data
                    }

                except Exception as ex:
                    self.exc = Exception(ex)

            def join(self):
                super().join()

                # Test exception attribute
                if self.exc:
                    raise self.exc

        # Query data
        qryer = []
        for eventinstrprm in eventinstrprm_list:
            qryer.append(Queryer(eventinstrprm))

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
        return self._get_table(Flag)


db_mngr = DatabaseManager()


class ConfigLinker:

    def __init__(self):

        self._cfg_mngr = instantiate_config_managers(
            [CfgParameter, CfgInstrType,
             CfgInstrument, CfgFlag
             ]
        )

    @property
    def cfg_mngr(self):
        """ """
        return self._cfg_mngr

    def get_parameters(self):
        """ """
        return self.cfg_mngr[PARAM_KEY].document

    def get_instr_types(self):
        """ """
        return self.cfg_mngr[INSTR_TYPE_KEY].document

    def get_instruments(self):
        """ """
        return self.cfg_mngr[INSTR_KEY].document

    def get_flags(self):
        """ """
        return self.cfg_mngr[FLAG_KEY].document


class EventManager:
    """Class for create an uniform event identifier"""

    def __init__(
        self, event_dt, instr_id, prm_abbr,
        batch_id, day_event, event_id=None,
    ):
        """

        Args:
            event_dt (str | datetime | pd.Timestamp):
            instr_id (str):
            prm_abbr (str):
            event_id (str | None):
            batch_id (str | None):
            day_event (bool | None):

        """

        # Set attributes
        self.event_dt = event_dt
        self._instr_id = instr_id
        self._prm_abbr = prm_abbr
        self._event_id = event_id
        self._batch_id = batch_id
        self._day_event = day_event

    @property
    def event_dt(self):
        return self._event_dt

    @event_dt.setter
    def event_dt(self, value):

        # Convert to datetime
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        elif isinstance(value, str):
            value = pd.to_datetime(value).to_pydatetime()

        # Check time zone (UTC)
        assert value.tzinfo == pytz.UTC

        self._event_dt = value

    @property
    def instr_id(self):
        return self._instr_id

    @property
    def prm_abbr(self):
        return self._prm_abbr

    @property
    def event_id(self):
        return self._event_id

    @property
    def batch_id(self):
        return self._batch_id

    @property
    def day_event(self):
        return self._day_event

    def __repr__(self):
        pp = pprint.PrettyPrinter()
        return pp.pformat(self.as_dict())

    def as_dict(self):
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
        """

        Args:
            event_list (list of EventManager):

        Returns:

        """
        event_list.sort()

    def get_ordered_list(self):
        return [self.event_dt, self.instr_id]

    def __eq__(self, other):
        return self.get_ordered_list() == other.get_ordered_list()

    def __ne__(self, other):
        return self.get_ordered_list() != other.get_ordered_list()

    def __lt__(self, other):
        return self.get_ordered_list() < other.get_ordered_list()

    def __le__(self, other):
        return self.get_ordered_list() <= other.get_ordered_list()

    def __gt__(self, other):
        return self.get_ordered_list() > other.get_ordered_list()

    def __ge__(self, other):
        return self.get_ordered_list() >= other.get_ordered_list()
