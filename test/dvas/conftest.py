"""
Copyright (c) 2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Database fixture for testing classes and function.

"""

# Import from python packages and modules
import pytest
from pytest_data import get_data

# Import from tested package
from dvas.database.database import DatabaseManager, DBAccess
from dvas.database.model import InstrType as TableInstrType
from dvas.database.model import Object as TableObject
from dvas.environ import path_var
from dvas.helper import AttrDict


#TODO
# Split into 2 fixtures. One for the DB reset and setup.
# And one for the data insertion
@pytest.fixture(autouse=True)
def db_init(request, tmp_path_factory):
    """Database init auto used fixture.

    Note:
        Use pytest_data package

    """
    db_data = get_data(
        request,
        'db_data',
        {'sub_dir': 'db_fixture_default', 'reset_db': True}
    )

    # Set db path
    path_var.local_db_path = tmp_path_factory.getbasetemp() / db_data['sub_dir']

    # Register db_path
    db_data.update({'db_path': path_var.local_db_path.as_posix()})

    # Set db
    db_mngr = DatabaseManager(reset_db=db_data['reset_db'])

    # Register db manager
    db_data.update({'db_mngr': db_mngr})

    # Insert data in db
    if 'data' in db_data.keys():

        data_out = []
        for i, arg in enumerate(db_data['data']):

            # Insert data
            try:
                db_mngr.add_data(**arg)

            # Create get object id only
            except Exception:
                # Get instr_type
                instr_type = db_mngr.get_or_none(
                    TableInstrType,
                    search={
                        'where': TableInstrType.type_name == arg[TableInstrType.type_name.name]
                    }
                )

                # Create instrument entry
                with DBAccess(db_mngr):
                    oid = TableObject.create(
                        srn=arg[TableObject.srn.name],
                        pid=arg[TableObject.pid.name],
                        instr_type=instr_type
                    ).oid

                db_data['data'][i].update({'oid': oid})

            data_out.append(arg)

        # Register data
        db_data.update({'data': data_out})

    return AttrDict(db_data)
