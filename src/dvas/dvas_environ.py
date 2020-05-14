"""
This module contains the package's environment variables.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import Python packages and module
import os
from pathlib import Path
from contextlib import contextmanager

# Import current package's modules
from .dvas_helper import SingleInstanceMetaClass
from .dvas_helper import TypedProperty

# Define package path
package_path = Path(__file__).parent


class TypedPropertyPath(TypedProperty):
    """Special typed property class for manage Path attributes"""

    def __init__(self):
        """Constructor"""
        super().__init__((Path, str))

    def __set__(self, instance, value):
        """Overwrite __set__ method"""
        if not isinstance(value, self._type):
            raise TypeError(f"Expected class {self._type}, got {type(value)}")

        # Convert to pathlib.Path
        value = Path(value)

        # Test OS compatibilty
        try:
            value.exists()
        except OSError:
            raise TypeError(f"{value} not valid OS path")
        instance.__dict__[self._name] = value


class GlobalPathVariablesManager(metaclass=SingleInstanceMetaClass):
    """Class to manage package's global directory path variables"""

    # Set class constant attributes
    _CST = [
        {'name': 'orig_data_path',
         'default': package_path / 'examples' / 'data',
         'os_nm': 'DVAS_ORIG_DATA_PATH'},
        {'name': 'config_dir_path',
         'default': package_path / 'examples' / 'config',
         'os_nm': 'DVAS_CONFIG_DIR_PATH'},
        {'name': 'local_db_path',
         'default': Path('.') / 'dvas_db',
         'os_nm': 'DVAS_LOCAL_DB_PATH'},
        {'name': 'output_path',
         'default': Path('.') / 'output',
         'os_nm': 'DVAS_OUTPUT_PATH'}
    ]

    # Define attributes
    orig_data_path = TypedPropertyPath()
    config_dir_path = TypedPropertyPath()
    local_db_path = TypedPropertyPath()
    output_path = TypedPropertyPath()

    def __init__(self):
        """Constructor"""
        # Load os environment values
        self.load_os_environ()

    def load_os_environ(self):
        """Load from OS environment variables"""
        for arg in self._CST:
            setattr(
                self,
                arg['name'],
                os.getenv(arg['os_nm'], arg['default'])
            )

    @contextmanager
    def set_many_attr(self, items):
        """Context manager to set temporarily many global variables.

        Args:
            items (dict): Temporarily global variables.
                keys: instance attribute name. values: temporarily set values.

        Examples:
            >>>from dvas.dvas_environ import path_var
            >>>with path_var.set_many_attr({})

        """

        # Set new values and save old values
        old_items = {}
        for key, val in items.items():
            old_items.update({key: getattr(self, key)})
            setattr(self, key, val)
        yield

        # Restore old values
        for key, val in old_items.items():
            setattr(self, key, val)

    @staticmethod
    def _test_path(value):
        """Test and convert input argument to pathlib.Path.

        If variable is not defined in environment variable, the default value
        is used.

        Args:
            value (`obj`): Argument to be tested

        """
        try:
            # Set Path
            out = Path(value)

            # Test OS compatibility
            out.exists()

            return out

        except (TypeError, OSError):
            raise TypeError('Not compatible with system path')


path_var = GlobalPathVariablesManager()
