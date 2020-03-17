"""

"""

# Import from python packages
import os
from pathlib import Path
import re
from peewee import SqliteDatabase, Model, Check
from peewee import AutoField
from peewee import IntegerField, BooleanField, FloatField
from peewee import DateTimeField, TextField, CharField
from peewee import ForeignKeyField, CompositeKey
from playhouse.hybrid import hybrid_property
from playhouse.shortcuts import fn


# Import from current package
from ..dvas_environ import LOCAL_DB_PATH_NM
from ..config.pattern import EVENT_PAT, BATCH_PAT
from ..config.pattern import INSTR_TYPE_PAT, INSTR_PAT
from ..config.pattern import PARAM_PAT

# Define
DATABASE_PATH = os.getenv(LOCAL_DB_PATH_NM)
DATABASE_FILE_PATH = Path(DATABASE_PATH) / 'local_db.sqlite'

db = SqliteDatabase(
    DATABASE_FILE_PATH,
    pragmas={
        'foreign_keys': True,

        # Set cache to 10MB
        'cache_size': -10*1024
    },
    autoconnect=False)


@db.func('re_fullmatch')
def re_fullmatch(pattern, string):
    """Database re.fullmatch function used in Check constraints"""
    return (
        re.fullmatch(pattern=pattern, string=string) is not None
    )


class MetadataModel(Model):
    """ """
    class Meta:
        database = db


class InstrType(MetadataModel):
    """ """
    id = AutoField(primary_key=True)
    name = CharField(
        null=False, unique=True,
        constraints=[
            Check(f"re_fullmatch('{INSTR_TYPE_PAT}', name)")
        ]
    )
    desc = TextField()


class Instrument(MetadataModel):
    """ """
    id = AutoField(primary_key=True)
    instr_id = CharField(
        constraints=[
            Check(f"re_fullmatch('{INSTR_PAT}', instr_id)")
        ]
    )
    instr_type = ForeignKeyField(InstrType, backref='instruments')
    sn = TextField()
    remark = TextField()


class Parameter(MetadataModel):
    """ """
    id = AutoField(primary_key=True)
    prm_abbr = CharField(
        null=False,
        unique=True,
        constraints=[
            Check(f"re_fullmatch('{PARAM_PAT}', prm_abbr)"),
        ]
    )
    prm_desc = TextField(default='')


class Flag(MetadataModel):
    """ """
    id = AutoField(primary_key=True)
    bit_number = IntegerField(
        null=False,
        unique=True,
        constraints=[Check("bit_number >= 0")])
    desc = TextField(null=False)


class OrgiDataInfo(MetadataModel):
    id = AutoField(primary_key=True)
    source = CharField(null=True)


class EventsInstrumentsParameters(MetadataModel):
    """ """
    id = AutoField(primary_key=True)
    event_dt = DateTimeField(null=False)
    instrument = ForeignKeyField(Instrument, backref='event_instrs_params')
    param = ForeignKeyField(Parameter, backref='event_instrs_params')
    event_id = CharField(null=True)
    batch_id = CharField(null=True)
    day_event = BooleanField(null=True)
    orig_data_info = ForeignKeyField(OrgiDataInfo, backref='event_instrs_params')


class Data(MetadataModel):
    """ """
    id = AutoField(primary_key=True)
    event_instr_param = ForeignKeyField(
        EventsInstrumentsParameters,
        backref='datas')
    index = FloatField(null=False)
    value = FloatField()
