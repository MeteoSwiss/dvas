"""
This file contains testing classes and function for
dvas.dvas_helper module.

"""

# Import from python packages and modules
import os
from pathlib import Path
import pytest

# Import from current package
from dvas.dvas_environ import path_var
from dvas.dvas_environ import GlobalPathVariablesManager


class TestGlobalPathVariablesManager:
    """Class to test GlobalPathVariablesManager"""

    # Define
    test_value = Path('.', 'my_test_dir')
    test_env_value = Path('my_env_test_dir')
    bad_test_value = Path('1+*%')
    attr_name = path_var.CST[0]['name']
    os_varname = path_var.CST[0]['os_nm']
    init_value = getattr(path_var, attr_name)
    path_var_2 = GlobalPathVariablesManager()

    def test_uniqueness(self):
        """Test instance uniqueness"""

        assert id(path_var) == id(self.path_var_2)

    def test_direct_assignment(self):
        """Method direct variable assignment

        The method tests:
            - pathlib.Path and str assignement
            - TypeError exception

        """

        # Test pathlib.Path value
        with path_var.set_many_attr({self.attr_name: self.test_value}):
            assert getattr(self.path_var_2, self.attr_name) == self.test_value

        print(getattr(self.path_var_2, self.attr_name))
        with path_var.set_many_attr(
                {self.attr_name: self.test_value.as_posix()}
        ):
            assert getattr(self.path_var_2, self.attr_name) == self.test_value

        # Test exception
        print(getattr(self.path_var_2, self.attr_name))
        with pytest.raises(TypeError):
            setattr(path_var, self.attr_name, self.bad_test_value)
        with pytest.raises(TypeError):
            setattr(path_var, self.attr_name, self.bad_test_value.as_posix())

    def test_misc(self):
        """Method to test miscellaneous behaviours.

        The method tests:
            - Load from OS environment variables
            - Temporarily set with context manager

        """

        # Init
        old_os_value = os.getenv(self.os_varname)
        os.environ[self.os_varname] = self.test_env_value.as_posix()

        # Test
        with path_var.set_many_attr({self.attr_name: self.test_value}):
            # Test direct assignment
            assert getattr(path_var, self.attr_name) == self.test_value
            assert getattr(self.path_var_2, self.attr_name) == self.test_value

            # Reload path var environ
            path_var.load_os_environ()

            # Test load from OS environ
            assert getattr(path_var, self.attr_name) == self.test_env_value
            assert getattr(self.path_var_2, self.attr_name) == self.test_env_value

        # Test correct exit from context manager
        assert getattr(path_var, self.attr_name) == self.init_value
        assert getattr(self.path_var_2, self.attr_name) == self.init_value

        # Set old OS environ value
        if old_os_value is not None:
            os.environ[self.os_varname] = old_os_value
