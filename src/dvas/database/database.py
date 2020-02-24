

from functools import wraps

from playhouse.shortcuts import model_to_dict

import pandas as pd


from .model import db, PARAMETER_NAMES, FLAG_NAMES
from .model import Instrument, InstrumentType, FlightsInstruments
from .model import Flight, Parameter, FlagBits, Data


def db_context(verbose=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            # Test class
            assert isinstance(args[0], DatabaseManager)

            try:
                with db.connection_context():
                    return func(*args, **kwargs)
            except Exception as e:
                if verbose:
                    #TODO
                    # Replace print by log???
                    print(f"Unsuccessful execution of {func.__name__}(*{args[1:]}, **{kwargs}): {e}")

        return wrapper
    return decorator


class DatabaseManager:
    _INSTANCES = 0

    def __init__(self):

        if DatabaseManager._INSTANCES > 0:
            raise Exception(f'More than one instance of class {self.__class__.__name__} has been created')
        else:
            DatabaseManager._INSTANCES += 1

        self._create_tables()
        self._create_parameters()
        self._create_flag()

    @db_context()
    def add_instr_type(self, name, desc=''):
        InstrumentType.get_or_create(name=name, desc=desc)

    @db_context()
    def del_instr_type(self, name):
        raise NotImplementedError(f"Please implements {self.__class__.del_instr_type}")

    @db_context()
    def add_instr(self, instr_id, instr_type_name, remark=''):
        instr_type = InstrumentType.get_or_none(InstrumentType.name == instr_type_name)
        if instr_type:
            Instrument.get_or_create(id=instr_id, instr_type=instr_type, remark=remark)

    @db_context()
    def add_flight(self, flight_id, launch_dt, day_flight, batch_amount, instr_id):

        instr = Instrument.get_or_none(Instrument.id == instr_id)
        if not instr:
            raise Exception(f'{instr_id} not found')

        if isinstance(launch_dt, pd.Timestamp):
            launch_dt = launch_dt.to_pydatetime()

        Flight.get_or_create(
            id=flight_id, launch_dt=launch_dt,
            day_flight=day_flight, batch_amount=batch_amount,
            ref_instr=instr
        )

    @db_context()
    def link_instr_flight(self, flight_id, instr_id, batch_id):

        flight = Flight.get_or_none(Flight.id == flight_id)
        if not flight:
            raise Exception(f'{flight_id} not found')

        instr = Instrument.get_or_none(Instrument.id == instr_id)
        if not instr:
            raise Exception(f'{instr_id} not found')

        FlightsInstruments.get_or_create(flight=flight, instrument=instr, batch_id=batch_id)

        # Check
        flight_instr = FlightsInstruments.get(flight=flight, instrument=instr, batch_id=batch_id)
        if flight_instr.batch_id_no >= flight.batch_amount:
            db.rollback()
            raise Exception(f"batch_id number must be < {flight.batch_amount}")

    @db_context()
    def get_instr(self, instr_id=None):
        if instr_id:
            qry = Instrument.select().where(Instrument.id.in_(instr_id))
        else:
            qry = Instrument.select()
        return [model_to_dict(arg) for arg in qry]

    @db_context()
    def get_flight(self, flight_id=None):
        if flight_id:
            qry = Flight.select().where(Flight.id.in_(flight_id))
        else:
            qry = Flight.select()
        return [model_to_dict(arg) for arg in qry]

    @db_context()
    def get_flight_instr(self, flight_id=None, instr_id=None):
        if (flight_id is None) and (instr_id is None):
            qry = FlightsInstruments.select()
        elif instr_id is None:
            qry = (
                FlightsInstruments
                .select()
                .join(Flight)
                .where(
                    Flight.id.in_(flight_id))
            )
        elif flight_id is None:
            qry = (
                FlightsInstruments
                .select()
                .join(Instrument)
                .where(
                    Instrument.id.in_(instr_id)
                )
            )
        else:
            qry = (
                FlightsInstruments
                .select()
                .join(Instrument)
                .switch(FlightsInstruments)
                .join(Flight)
                .where(
                    Instrument.id.in_(instr_id) & Flight.id.in_(flight_id)
                )
            )
        return [model_to_dict(arg) for arg in qry]

    @db_context()
    def get_parameter(self, parameter_id=None):
        if parameter_id:
            qry = Parameter.select().where(Parameter.id.in_(parameter_id))
        else:
            qry = Parameter.select()
        return [model_to_dict(arg) for arg in qry]

    @db_context()
    def _create_tables(self):
        db.create_tables(
            [Flight, InstrumentType, Instrument,
             FlightsInstruments, Parameter, FlagBits,
             Data]
        )

    @db_context(False)
    def _create_parameters(self):
        Parameter.insert_many(PARAMETER_NAMES).execute()

    @db_context(False)
    def _create_flag(self):
        FlagBits.insert_many(FLAG_NAMES).execute()

    def print(self):

        print(Instrument.__name__)
        for arg in self.get_instr():
            print(arg)

        print()

        print(Flight.__name__)
        for arg in self.get_flight():
            print(arg)

        print()

        print(FlightsInstruments.__name__)
        for arg in self.get_flight_instr():
            print(arg)

        print()

        print(Parameter.__name__)
        for arg in self.get_parameter():
            print(arg)


db_mngr = DatabaseManager()
