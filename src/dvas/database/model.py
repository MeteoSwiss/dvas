"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Database model (ORM uses PeeWee package)

"""

# Import from python packages
import re
from peewee import SqliteDatabase, Model, Check
from peewee import AutoField
from peewee import IntegerField, FloatField
from peewee import DateTimeField, TextField, CharField
from peewee import ForeignKeyField

# Import from current package
from ..config.pattern import INSTR_TYPE_PAT
from ..config.pattern import PARAM_PAT


# Create db instance
db = SqliteDatabase(None, autoconnect=True)


@db.func('re_fullmatch')
def re_fullmatch(pattern, string):
    """Database re.fullmatch function. Used it in check constraints"""
    return (
        re.fullmatch(pattern=pattern, string=string) is not None
    )


class MetadataModel(Model):
    """Metadata model class"""
    class Meta:
        """Meta class"""
        database = db


class InstrType(MetadataModel):
    """Instrument type model"""
    id = AutoField(primary_key=True)
    type_name = CharField(
        null=True, unique=True,
        constraints=[
            Check(f"re_fullmatch('({INSTR_TYPE_PAT})|()', type_name)")
        ]
    )
    desc = TextField()


class Instrument(MetadataModel):
    """Instrument model """
    id = AutoField(primary_key=True)
    sn = CharField(null=True, unique=True)
    instr_type = ForeignKeyField(InstrType, backref='instruments')
    remark = TextField()


class Parameter(MetadataModel):
    """Parameter model"""
    id = AutoField(primary_key=True)
    prm_abbr = CharField(
        null=False,
        unique=True,
        constraints=[
            Check(f"re_fullmatch('{PARAM_PAT}', prm_abbr)"),
        ]
    )
    prm_desc = TextField(null=False, default='')


class Flag(MetadataModel):
    """Flag model"""
    id = AutoField(primary_key=True)
    bit_number = IntegerField(
        null=False,
        unique=True,
        constraints=[Check("bit_number >= 0")])
    flag_abbr = CharField(null=False)
    flag_desc = TextField(null=False, default='')


class Tag(MetadataModel):
    """Tag model"""
    id = AutoField(primary_key=True)
    tag_abbr = CharField(null=False, unique=True)
    tag_desc = TextField()


class OrgiDataInfo(MetadataModel):
    """Original data information model"""
    id = AutoField(primary_key=True)
    source = CharField(null=True)


class EventsInfo(MetadataModel):
    """Events/Instruments/Parameter model"""
    id = AutoField(primary_key=True)
    event_dt = DateTimeField(null=False)
    instrument = ForeignKeyField(
        Instrument, backref='event_info'
    )
    param = ForeignKeyField(
        Parameter, backref='event_info'
    )
    orig_data_info = ForeignKeyField(
        OrgiDataInfo, backref='event_info'
    )


class EventsTags(MetadataModel):
    """Many-to-Many link between EventsInfo and Tag tables"""
    tag = ForeignKeyField(
        Tag, backref='events_tags'
    )
    events_info = ForeignKeyField(
        EventsInfo, backref='events_tags'
    )


#TODO
#class MetaData
#    prm
#    value
#    event


class Data(MetadataModel):
    """Data model"""
    id = AutoField(primary_key=True)
    event_info = ForeignKeyField(
        EventsInfo,
        backref='datas')
    index = IntegerField(null=False)
    value = FloatField(null=True)
