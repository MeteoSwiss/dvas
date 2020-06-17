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
from dvas.dvas_environ import set_path
from dvas.dvas_environ import GlobalPathVariablesManager


@pytest.fixture(autouse=True)
def test_set_path(tmpdir):
    """Function to test set_path"""

    # Test str path name
    assert set_path(Path(tmpdir).as_posix()) == Path(tmpdir)

    # Test exist_ok True
    assert set_path(Path(tmpdir), exist_ok=True) == Path(tmpdir)

    # Test exist_ok False
    assert set_path(Path(tmpdir) / 'test') == Path(tmpdir) / 'test'

    # Raise exception
    with pytest.raises(TypeError):
        set_path(Path(tmpdir) / 'dummy', exist_ok=True)


class TestGlobalPathVariablesManager:
    """Class to test GlobalPathVariablesManager"""

    # Define
    ok_value = Path('my_test_dir', 'my_sub_dir')
    ok_env_value = Path('my_env_test_dir', 'my_sub_env_dir')
    attr_name = 'output_path'
    os_varname = 'DVAS_OUTPUT_PATH'
    init_value = getattr(path_var, attr_name)
    path_var_2 = GlobalPathVariablesManager()

    def test_uniqueness(self):
        """Test instance uniqueness"""
        assert id(path_var) == id(self.path_var_2)

    @pytest.fixture(autouse=True)
    def test_direct_assignment(self, tmpdir):
        """Method direct variable assignment

        The method tests:
            - pathlib.Path and str assignement
            - TypeError exception

        """

        # Test pathlib.Path value
        test_value = Path(tmpdir) / self.ok_value
        with path_var.set_many_attr({self.attr_name: test_value}):
            assert getattr(self.path_var_2, self.attr_name) == test_value

        with path_var.set_many_attr(
                {self.attr_name: test_value.as_posix()}
        ):
            assert getattr(self.path_var_2, self.attr_name) == test_value

    @pytest.fixture(autouse=True)
    def test_misc(self, tmpdir):
        """Method to test miscellaneous behaviours.

        The method tests:
            - Load from OS environment variables
            - Temporarily set with context manager

        """

        # Init
        old_os_value = os.getenv(self.os_varname)
        test_value = Path(tmpdir) / self.ok_env_value
        os.environ[self.os_varname] = test_value.as_posix()

        # Test
        with path_var.set_many_attr({self.attr_name: self.init_value}):

            # Reload path var environ
            path_var.load_os_environ()

            # Test load from OS environ
            assert getattr(path_var, self.attr_name) == test_value
            assert getattr(self.path_var_2, self.attr_name) == test_value

        # Set old OS environ value
        if old_os_value is not None:
            os.environ[self.os_varname] = old_os_value
