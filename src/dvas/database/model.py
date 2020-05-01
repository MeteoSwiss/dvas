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
from playhouse.sqliteq import SqliteQueueDatabase


# Import from current package
from ..dvas_environ import LOCAL_DB_PATH_NM
from ..config.pattern import EVENT_PAT, BATCH_PAT
from ..config.pattern import INSTR_TYPE_PAT, INSTR_PAT
from ..config.pattern import PARAM_PAT

# Define
DATABASE_PATH = os.getenv(LOCAL_DB_PATH_NM)
DATABASE_FILE_PATH = Path(DATABASE_PATH) / 'local_db.sqlite'

#SqliteDatabase
#SqliteQueueDatabase
db = SqliteQueueDatabase(
    DATABASE_FILE_PATH,
    pragmas={
        'foreign_keys': True,
        # Set cache to 10MB
        'cache_size': -10*1024
    },
    autoconnect=False,
    use_gevent=False,  # Use the standard library "threading" module.
    autostart=False,  # The worker thread now must be started manually.
    queue_max_size=64,  # Max. # of pending writes that can accumulate.
    results_timeout=5.0  # Max. time to wait for query to be executed.
)


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
    type_name = CharField(
        null=False, unique=True,
        constraints=[
            Check(f"re_fullmatch('{INSTR_TYPE_PAT}', type_name)")
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
    flag_abbr = CharField(null=False)
    desc = TextField(null=False)


class OrgiDataInfo(MetadataModel):
    id = AutoField(primary_key=True)
    source = CharField(null=True)


class EventsInstrumentsParameters(MetadataModel):
    """ """
    id = AutoField(primary_key=True)
    event_dt = DateTimeField(null=False)
    instrument = ForeignKeyField(
        Instrument, backref='event_instrs_params'
    )
    param = ForeignKeyField(
        Parameter, backref='event_instrs_params'
    )
    batch_id = CharField(null=False)
    day_event = BooleanField(null=False)
    event_id = CharField(null=True)
    orig_data_info = ForeignKeyField(
        OrgiDataInfo, backref='event_instrs_params'
    )


class Data(MetadataModel):
    """ """
    id = AutoField(primary_key=True)
    event_instr_param = ForeignKeyField(
        EventsInstrumentsParameters,
        backref='datas')
    index = FloatField(null=False)
    value = FloatField(null=True)
