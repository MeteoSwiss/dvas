"""
Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Database model (ORM uses PeeWee package)

"""

# Import from python packages
import logging
import re
from datetime import datetime
import pandas as pd
from peewee import SqliteDatabase, Check
from peewee import Model as PeeweeModel
from peewee import AutoField, Field
from peewee import IntegerField, FloatField, TextField
from peewee import ForeignKeyField

# Import from current package
from ..hardcoded import MODEL_PAT, PRM_AND_FLG_PRM_PAT

# Setup local logger
logger = logging.getLogger(__name__)

# Create db instance
db = SqliteDatabase(None, autoconnect=True)


@db.func('re_fullmatch')
def re_fullmatch(pattern, string):
    """ Database re.fullmatch function. Used in check constraints. """
    return (
        re.fullmatch(pattern=pattern, string=string) is not None
    )


@db.func('str_len_max')
def str_len_max(string, n_max):
    """ Database string length max function. Used in check constraints """
    if string is None:
        out = True
    else:
        out = len(string) <= n_max
    return out


@db.func('check_unit')
def check_unit(prm_unit, prm_name):
    """ Apply verification checks on the units """

    # This check is not robust, in the sense that users have the freedom to choose whatever
    # parameter name they prefer. But it is better than nothing.
    # If a time is specified, let's make sure I can use the unit to convert this to timedeltas via
    # pandas (will be required when creating Profiles).
    if prm_name in ['time']:
        try:
            pd.to_timedelta(pd.Series([0, 1]), prm_unit)

            # For now, we force the use of 's' when extracting data from the db.
            # Until this changes, let's raise an error if this is not what the user provided.
            # See #192 and data.startegy.data.py for details.
            if prm_unit != 's':
                msg = 'Only "s" is allowed as the unit of time data.'
                msg += ' See Github error #192 for details.'
                logger.error(msg)
                return False
            # ---------------------------------
            return True
        except ValueError:
            logger.error('Unknown unit for parameter %s: %s', prm_name, prm_unit)
            return False

    return True


class TimestampTZField(Field):
    """
    A timestamp field that supports a timezone by serializing the value
    with isoformat.

    Source: Justin Turpin, https://compileandrun.com/python-peewee-timezone-aware-datetime/
    """

    field_type = 'TEXT'  # This is how the field appears in Sqlite

    def db_value(self, value: datetime) -> str:
        if value:
            return value.isoformat()
        return None

    def python_value(self, value: str) -> str:
        if value:
            return datetime.fromisoformat(value)
        return None


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
            Check("str_len_max(mdl_name, 64)")
        ]
    )

    # Model description
    mdl_desc = TextField(
        null=True, unique=False, default='',
        constraints=[Check("str_len_max(mdl_desc, 256)")]
    )

    # Model identifier
    mid = TextField(
        null=False, unique=False, default='',
        constraints=[Check("str_len_max(mid, 64)")]
    )


class Object(MetadataModel):
    """Object table"""

    # Object id
    oid = AutoField(primary_key=True)

    # Object serial number
    srn = TextField(
        null=False,
        constraints=[Check("str_len_max(srn, 64)")]
    )

    # Object product identifier
    pid = TextField(
        null=False,
        constraints=[Check("str_len_max(pid, 64)")]
    )

    # Link to model
    model = ForeignKeyField(
        Model, backref='objects', on_delete='CASCADE'
    )


class Prm(MetadataModel):
    """Parameter model"""

    # Table id
    prm_id = AutoField(primary_key=True)

    # Parameter name
    prm_name = TextField(
        null=False,
        unique=True,
        constraints=[Check(f"re_fullmatch('{PRM_AND_FLG_PRM_PAT}', prm_name)"),
                     Check("str_len_max(prm_name, 64)")]
    )
    # Parameter ame for plots
    prm_plot = TextField(
        null=False,
        #unique=True,
        constraints=[Check("str_len_max(prm_name, 64)")]
    )
    # Parameter description
    prm_desc = TextField(
        null=False, default='',
        constraints=[Check("str_len_max(prm_desc, 256)")]
    )

    # Parameter comment
    prm_cmt = TextField(
        null=False, default='',
        constraints=[Check("str_len_max(prm_cmt, 256)")]
    )

    # Parameter units
    prm_unit = TextField(
        null=False, default='',
        constraints=[Check("str_len_max(prm_unit, 64)"),
                     Check("check_unit(prm_unit, prm_name)")]
    )


class Flg(MetadataModel):
    """Flag model"""

    # Table id
    flg_id = AutoField(primary_key=True)

    # Bit position
    bit_pos = IntegerField(
        null=False,
        unique=True,
        # Can only go up to 62, because we need one bit to store the sign
        constraints=[Check("bit_pos >= 0"), Check("bit_pos <= 62")])

    # Flag name
    flg_name = TextField(
        null=False, unique=True,
        constraints=[Check("str_len_max(flg_name, 64)")]
    )

    # Flag description
    flg_desc = TextField(
        null=False, default='',
        constraints=[Check("str_len_max(flg_desc, 256)")]
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
        constraints=[Check("str_len_max(tag_name, 64)")]
    )

    # Tag description
    tag_desc = TextField(
        null=True, unique=False, default='',
        constraints=[Check("str_len_max(tag_desc, 256)")]
    )


class DataSource(MetadataModel):
    """Data source model"""
    id = AutoField(primary_key=True)
    src = TextField(
        null=False,
        constraints=[Check("str_len_max(src, 2048)")]
    )


class Info(MetadataModel):
    """Info table"""

    # Info id
    info_id = AutoField(primary_key=True)
    edt = TimestampTZField(null=False)
    param = ForeignKeyField(
        Prm, backref='info', on_delete='CASCADE'
    )
    data_src = ForeignKeyField(
        DataSource, backref='info', on_delete='CASCADE'
    )
    evt_hash = TextField(
        null=False,
        constraints=[Check("str_len_max(evt_hash, 64)")]
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
    """ Table containing the profiles metadata.

    Note:
        Metadata table should be used only to save metadata associated
        to a profile.

    """
    metadata_id = AutoField(primary_key=True)

    #: str: Metadata key name
    key_name = TextField(
        null=False,
        constraints=[Check("str_len_max(key_name, 64)")]
    )

    #: str: Metadata key string value
    value_str = TextField(
        null=True,
        constraints=[Check("str_len_max(value_str, 256)")]
    )

    #: float: Metadata key float value
    value_num = FloatField(null=True)

    #: datetime.datetime: Metadata key datetime value
    value_datetime = TimestampTZField(null=True)

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
