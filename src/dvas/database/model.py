"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Database model (ORM uses PeeWee package)

"""

# Import from python packages
import re
from peewee import SqliteDatabase, Check
from peewee import Model as PeeweeModel
from peewee import AutoField
from peewee import IntegerField, FloatField
from peewee import DateTimeField, TextField
from peewee import ForeignKeyField


# Import from current package
from ..config.pattern import MODEL_PAT
from ..config.pattern import PARAM_PAT


# Create db instance
db = SqliteDatabase(None, autoconnect=True)


@db.func('re_fullmatch')
def re_fullmatch(pattern, string):
    """Database re.fullmatch function. Used it in check constraints"""
    return (
        re.fullmatch(pattern=pattern, string=string) is not None
    )


@db.func('str_len_max')
def str_len_max(string, n_max):
    """Database string length max function. Used it in check constraints"""
    if string is None:
        out = True
    else:
        out = len(string) <= n_max
    return out


class MetadataModel(PeeweeModel):
    """Metadata model class"""
    class Meta:
        """Meta class"""
        database = db


class Model(MetadataModel):
    """Model table, intended as object model"""

    # Table id
    mdl_id = AutoField(primary_key=True)

    # Model name
    mdl_name = TextField(
        null=False, unique=True,
        constraints=[
            Check(f"re_fullmatch('({MODEL_PAT})|()', mdl_name)"),
            Check(f"str_len_max(mdl_name, 64)")
        ]
    )

    # Model description
    mdl_desc = TextField(
        null=True, unique=False, default='',
        constraints=[Check(f"str_len_max(mdl_desc, 256)")]
    )

    # Model identifier
    mid = TextField(
        null=False, unique=False, default='',
        constraints=[Check(f"str_len_max(mdl_desc, 64)")]
    )


class Object(MetadataModel):
    """Object table"""

    # Object id
    oid = AutoField(primary_key=True)

    # Object serial number
    srn = TextField(
        null=False,
        constraints=[Check(f"str_len_max(srn, 64)")]
    )

    # Object product identifier
    pid = TextField(
        null=False,
        constraints=[Check(f"str_len_max(pid, 64)")]
    )

    # Link to model
    model = ForeignKeyField(
        Model, backref='objects', on_delete='CASCADE'
    )


class Parameter(MetadataModel):
    """Parameter model"""

    # Table id
    prm_id = AutoField(primary_key=True)

    # Parameter name
    prm_name = TextField(
        null=False,
        unique=True,
        constraints=[
            Check(f"re_fullmatch('{PARAM_PAT}', prm_name)"),
            Check(f"str_len_max(prm_name, 64)")
        ]
    )

    # Parameter description
    prm_desc = TextField(
        null=False, default='',
        constraints=[Check(f"str_len_max(prm_desc, 256)")]
    )

    # Parameter units
    prm_unit = TextField(
        null=False, default='',
        constraints=[Check(f"str_len_max(prm_unit, 64)")]
)


class Flag(MetadataModel):
    """Flag model"""

    # Table id
    flag_id = AutoField(primary_key=True)

    # Bit position
    bit_pos = IntegerField(
        null=False,
        unique=True,
        constraints=[Check("bit_pos >= 0")])

    # Flag name
    flag_name = TextField(
        null=False, unique=True,
        constraints=[Check(f"str_len_max(flag_name, 64)")]
    )

    # Flag description
    flag_desc = TextField(
        null=False, default='',
        constraints=[Check(f"str_len_max(flag_desc, 256)")]
    )


class Tag(MetadataModel):
    """Table containing the tags.

    Note:
        Tags should be used to search profiles in the DB.

    """

    # Table id
    id = AutoField(primary_key=True)

    # Tag name
    tag_name = TextField(
        null=False, unique=True,
        constraints = [Check(f"str_len_max(tag_name, 64)")]
    )

    # Tag description
    tag_desc = TextField(
        null=True, unique=False, default='',
        constraints=[Check(f"str_len_max(tag_desc, 256)")]
    )


class DataSource(MetadataModel):
    """Data source model"""
    id = AutoField(primary_key=True)
    src = TextField(
        null=False,
        constraints=[Check(f"str_len_max(src, 2048)")]
    )


class Info(MetadataModel):
    """Info table"""

    # Info id
    info_id = AutoField(primary_key=True)
    edt = DateTimeField(null=False)
    param = ForeignKeyField(
        Parameter, backref='info', on_delete='CASCADE'
    )
    data_src = ForeignKeyField(
        DataSource, backref='info', on_delete='CASCADE'
    )
    evt_hash = TextField(
        null=False,
        constraints=[Check(f"str_len_max(evt_hash, 64)")]
    )
    """str: Hash of the info attributes. Using a hash allows you to manage
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


class InfosObjects(MetadataModel):
    """Many-to-Many link between Info and Instrument tables"""
    id = AutoField(primary_key=True)
    object = ForeignKeyField(
        Object, backref='infos_objects', on_delete='CASCADE'
    )
    info = ForeignKeyField(
        Info, backref='infos_objects', on_delete='CASCADE'
    )


class MetaData(MetadataModel):
    """Table containing the profiles metadata.

    Note:
        Metadata table should be used only to save metadata associated
        to a profile.

    """
    metadata_id = AutoField(primary_key=True)

    #: str: Metadata key name
    key_name = TextField(
        null=False,
        constraints=[Check(f"str_len_max(key_name, 64)")]
    )

    #: str: Metadata key string value
    value_str = TextField(
        null=True,
        constraints=[Check(f"str_len_max(value_str, 256)")]
    )

    #: float: Metadata key float value
    value_num = FloatField(null=True)

    #: peewee.Model: Link to Info table
    info = ForeignKeyField(
        Info, backref='infos_objects', on_delete='CASCADE'
    )


class Data(MetadataModel):
    """Table containing the profiles data."""
    id = AutoField(primary_key=True)
    info = ForeignKeyField(
        Info,
        backref='datas', on_delete='CASCADE')
    index = IntegerField(null=False)
    value = FloatField(null=True)
