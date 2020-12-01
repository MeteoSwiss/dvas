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

    # Serial number
    srn = CharField(null=True, unique=True)
    instr_type = ForeignKeyField(
        InstrType, backref='instruments', on_delete='CASCADE'
    )
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
    tag_txt = CharField(null=False, unique=True)
    tag_desc = TextField()


class DataSource(MetadataModel):
    """Data source model"""
    id = AutoField(primary_key=True)
    source = CharField(null=True)
    source_hash = IntegerField()


class Info(MetadataModel):
    """Info model"""
    id = AutoField(primary_key=True)
    evt_dt = DateTimeField(null=False)
    param = ForeignKeyField(
        Parameter, backref='info', on_delete='CASCADE'
    )
    data_src = ForeignKeyField(
        DataSource, backref='info', on_delete='CASCADE'
    )
    evt_hash = IntegerField()
    """int: Hash of the info attributes. Using a hash allows you to manage
    identical info with varying degrees of work steps."""


class InfosTags(MetadataModel):
    """Many-to-Many link between Info and Tag tables"""
    id = AutoField(primary_key=True)
    tag = ForeignKeyField(
        Tag, backref='infos_tags', on_delete='CASCADE'
    )
    info = ForeignKeyField(
        Info, backref='infos_tags', on_delete='CASCADE'
    )


class InfosInstruments(MetadataModel):
    """Many-to-Many link between Info and Instrument tables"""
    id = AutoField(primary_key=True)
    instr = ForeignKeyField(
        Instrument, backref='instruments_infos', on_delete='CASCADE'
    )
    info = ForeignKeyField(
        Info, backref='instruments_infos', on_delete='CASCADE'
    )

# TODO
#  Add the capability to link metadata to an info.
#  Tag should be used to search data in the DB.
#  Metadata should be used to save metadata of a result profile
#  class MetaData
#     prm
#     value
#     info

class Data(MetadataModel):
    """Data model"""
    id = AutoField(primary_key=True)
    info = ForeignKeyField(
        Info,
        backref='datas', on_delete='CASCADE')
    index = IntegerField(null=False)
    value = FloatField(null=True)
