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
from dvas.database.database import DatabaseManager
from dvas.database.model import Model as TableModel
from dvas.database.model import Object as TableObject
from dvas.environ import path_var
from dvas.helper import AttrDict


def pytest_addoption(parser):
    """ A nifty little function that allows to feed command line arguments to the pytest command,
    e.g.:

        pytest --latex

    Intended to enable the use of a local LateX installation when running tests locally (i.e. NOT on
    Github).
    """

    parser.addoption("--latex", action="store_true",
                     help="Test plots also using a full (local) LaTeX installation.")

    parser.addoption("--show-plots", action="store_true",
                     help="Display the test plots on screen.")


@pytest.fixture(scope='session')
def do_latex(request):
    """ A pytext fixture to identify whether a local LaTeX installation exists, or not.

    Adapted from the response of ipetrik on
    `StackOverflow <https://stackoverflow.com/questions/40880259>`__

    To use this, simply call it as an argument in any of the test function, e.g.:

        def test_some_func(a, b, do_latex):
            ...
            if do_latex:
                dpu.set_mplstyle('latex')
                some_test_plot(...)
    """

    try:
        return request.config.getoption("--latex")
    except ValueError:
        # This happens if pytest is being launched from the root directory, in which case it does
        # NOT see conftest.py.
        return False

@pytest.fixture(scope='session')
def show_plots(request):
    """ A pytext fixture to decide whether test plots should be shown on-screen, or not.

    This is useful to disable all displays when running automated tests using Github actions,
    which, on some OS, do not handle the display of plots very well.

    To use this, simply call it as an argument in any of the test function, e.g.:

        def test_some_func(a, b, show_plot):
            ...
            some_test_plot(..., show=show_plot, ...)
    """

    try:
        return request.config.getoption("--show-plots")
    except ValueError:
        # This happens if pytest is being launched from the root directory, in which case it does
        # NOT see conftest.py.
        return False



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
    DatabaseManager().refresh_db()
    db_mngr = DatabaseManager()

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
                # Get model
                model = db_mngr.get_or_none(
                    TableModel,
                    search={
                        'where': TableModel.mdl_name == arg[TableModel.mdl_name.name]
                    }
                )

                # Create instrument entry
                oid = TableObject.create(
                    srn=arg[TableObject.srn.name],
                    pid=arg[TableObject.pid.name],
                    model=model
                ).oid

                #TODO: shouldn't the following line be:
                # arg.update({'oid': oid})
                # ? fpavogt, 16.06.2021
                db_data['data'][i].update({'oid': oid})

            data_out.append(arg)

        # Register data
        db_data.update({'data': data_out})

    return AttrDict(db_data)
