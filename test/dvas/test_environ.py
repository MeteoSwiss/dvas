"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.dvas_environ module.

"""

# Import from python packages and modules
import os
from pathlib import Path
import pytest

# Import from current package
from dvas.environ import path_var
from dvas.environ import GlobalPathVariablesManager


class TestGlobalPathVariablesManager:
    """Class to test GlobalPathVariablesManager"""

    # Define
    ok_value = Path('my_test_dir', 'my_sub_dir')
    ok_env_value = Path('my_env_test_dir', 'my_sub_env_dir')
    attr_test = 'output_path'
    os_varname = 'DVAS_OUTPUT_PATH'
    init_value = getattr(path_var, attr_test)
    path_var_2 = GlobalPathVariablesManager()

    def test_uniqueness(self):
        """Test instance uniqueness"""
        assert id(path_var) == id(self.path_var_2)

    @pytest.fixture(autouse=True)
    def test_direct_assignment(self, tmpdir):
        """Method direct variable assignment

        The method tests:
            - pathlib.Path and str assignment
            - TypeError exception

        """

        # Test pathlib.Path value
        test_value = Path(tmpdir) / self.ok_value
        with path_var.protect():
            setattr(path_var, self.attr_test, test_value)
            assert getattr(self.path_var_2, self.attr_test) == test_value

        with path_var.protect():
            setattr(path_var, self.attr_test, test_value.as_posix())
            assert getattr(self.path_var_2, self.attr_test) == test_value

    @pytest.fixture(autouse=True)
    def test_misc(self, tmpdir):
        """Method to test miscellaneous behaviours.

        The method tests:
            - Load from OS environment variables
            - Temporarily set with context manager

        """

        # Init
        test_value = Path(tmpdir) / self.ok_env_value
        os.environ[self.os_varname] = test_value.as_posix()

        # Test
        with path_var.protect():

            # Reload attr from environ
            path_var.set_default_attr()

            # Test load from OS environ
            assert getattr(path_var, self.attr_test) == test_value
            assert getattr(self.path_var_2, self.attr_test) == test_value
