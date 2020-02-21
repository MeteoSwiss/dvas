
from pathlib import Path
import re
from abc import ABC, abstractmethod, ABCMeta

from peewee import SqliteDatabase, Model, Check
from peewee import AutoField
from peewee import IntegerField, BooleanField, FloatField
from peewee import DateTimeField, TextField, CharField
from peewee import ForeignKeyField, CompositeKey
from playhouse.hybrid import hybrid_property
from playhouse.shortcuts import fn
from peewee import Expression, SQL
from playhouse.shortcuts import model_to_dict

DATABASE_FILE = Path('local_db.sqlite')

INSTR_NAME_LEN_MAX = 10
PRM_NAME_LEN_MAX = 8
BATCH_NUMBER_MAX = 2

FLIGHT_PREFIX = 'f'
FLIGHT_PATTERN = rf"{FLIGHT_PREFIX}(\d\d)"

INSTR_PREFIX = 'i'
INSTR_PATTERN = rf"{INSTR_PREFIX}(\d\d)"

BATCH_PREFIX = 'b'
BATCH_PATTERN = rf"{BATCH_PREFIX}(\d)"


db = SqliteDatabase(
    DATABASE_FILE,
    pragmas={'foreign_keys': 1},
    autoconnect=False)


PARAMETER_NAMES = [
    {'name': 'trepros1', 'desc': 'Temperature'},
    {'name': 'urepros1', 'desc': 'Humidity'},
    {'name': 'prepros1', 'desc': 'Pressure'},
    {'name': 'altpros1', 'desc': 'Altitude'},
    {'name': 'fklpros1', 'desc': 'Wind speed'},
    {'name': 'dklpros1', 'desc': 'Wind direction'},
    {'name': 'treprosu', 'desc': 'Temperature uncertainty'},
    {'name': 'ureprosu', 'desc': 'Humidity uncertainty'},
    {'name': 'preprosu', 'desc': 'Pressure uncertainty'},
    {'name': 'altprosu', 'desc': 'Altitude uncertainty'},
    {'name': 'fklprosu', 'desc': 'Wind speed uncertainty'},
    {'name': 'dklprosu', 'desc': 'Wind direction uncertainty'},
]

FLAG_NAMES = [
    {'bit': 0, 'desc': 'raw'},
    {'bit': 1, 'desc': 'raw NA'},
    {'bit': 2, 'desc': 'interpolated'},
    {'bit': 3, 'desc': 'synchronized'},
    {'bit': 4, 'desc': 'auto QC flag'},
    {'bit': 5, 'desc': 'manually QC flag'}
]

@db.func('rematch')
def rematch(pattern, string):
    """Database re.match function used in Check constraints"""
    return re.match(pattern=pattern, string=string) is not None


@db.func('re_full_match')
def re_full_match(pattern, string):
    """Database re.fullmatch function used in Check constraints"""
    return re.fullmatch(pattern=pattern, string=string) is not None


class MetadataModel(Model):
    class Meta:
        database = db


class InstrumentType(MetadataModel):
    _id = AutoField(primary_key=True)
    name = CharField(
        null=False, unique=True,
        constraints=[
            Check(f'length(name) <= {INSTR_NAME_LEN_MAX}')
        ]
    )
    desc = TextField()


class Instrument(MetadataModel):
    _id = AutoField(primary_key=True)
    id = CharField(
        constraints=[
            Check(f"re_full_match('{INSTR_PATTERN}', id)")
        ]
    )
    instr_type = ForeignKeyField(InstrumentType, backref='instruments')
    remark = TextField()


class Flight(MetadataModel):
    _id = AutoField(primary_key=True)
    id = CharField(
        constraints=[
            Check(f"re_full_match('{FLIGHT_PATTERN}', id)")
        ]
    )
    day_flight = BooleanField(null=False)
    launch_dt = DateTimeField(null=False)
    batch_amount = IntegerField(
        default=1,
        constraints=[
            Check('batch_amount > 0'),
            Check(f'batch_amount <= {BATCH_NUMBER_MAX}')
        ]
    )
    ref_instr = ForeignKeyField(Instrument, null=False)


class FlightsInstruments(MetadataModel):
    flight = ForeignKeyField(Flight, backref='flight_instruments')
    instrument = ForeignKeyField(Instrument, backref='flight_instruments')
    batch_id = CharField(
        null=False,
        constraints=[
            Check(f"re_full_match('{BATCH_PATTERN}', batch_id)")
        ]
    )

    @hybrid_property
    def batch_id_no(self):
        return int(re.match(BATCH_PATTERN, self.batch_id)[1])

    class Meta:
        primary_key = CompositeKey('flight', 'instrument')


class Parameter(MetadataModel):
    _id = AutoField(primary_key=True)
    name = CharField(null=False, unique=True, constraints=[Check(f'length(name) <= {PRM_NAME_LEN_MAX}')])
    desc = TextField(default='')


class FlagBits(MetadataModel):
    _id = AutoField(primary_key=True)
    bit = IntegerField(null=False, unique=True)
    desc = CharField(null=False)


class Data(MetadataModel):
    flight = ForeignKeyField(Flight, backref='datas')
    instrument = ForeignKeyField(Instrument, backref='datas')
    param = ForeignKeyField(Parameter, backref='datas')
    rel_time = FloatField(null=False)
    value = FloatField(null=False)
    flag = IntegerField(null=False)

    class Meta:
        primary_key = CompositeKey('flight', 'instrument', 'param', 'rel_time')
