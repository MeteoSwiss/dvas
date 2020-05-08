"""
This module contains the package's environment variables.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import Python packages and module
import os
from pathlib import Path

# Import current package's modules
from .dvas_helper import SingleInstanceMetaClass

# Define package path
package_path = Path(__file__).parent


class GlobalPathVariablesManager(metaclass=SingleInstanceMetaClass):
    """Class to manage global path package's variables"""

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

    def __init__(self):
        # Create private attributes
        self._orig_data_path = None
        self._config_dir_path = None
        self._local_db_path = None
        self._output_path = None

        # Load os environment values
        self.load_os_environ()

    @property
    def orig_data_path(self):
        """pathlib.Path: Original data directory path"""
        return self._orig_data_path

    @orig_data_path.setter
    def orig_data_path(self, value):
        self._orig_data_path = self._test_path(value)

    @property
    def config_dir_path(self):
        """pathlib.Path: Config directory path"""
        return self._config_dir_path

    @config_dir_path.setter
    def config_dir_path(self, value):
        self._config_dir_path = self._test_path(value)

    @property
    def local_db_path(self):
        """pathlib.Path: Local DB directory path"""
        return self._local_db_path

    @local_db_path.setter
    def local_db_path(self, value):
        self._local_db_path = self._test_path(value)

    @property
    def output_path(self):
        """pathlib.Path: Output directory path"""
        return self._output_path

    @output_path.setter
    def output_path(self, value):
        self._output_path = self._test_path(value)

    def load_os_environ(self):
        """Load from OS environment variables"""
        for arg in self._CST:
            setattr(
                self,
                arg['name'],
                os.getenv(arg['os_nm'], arg['default'])
            )

    @staticmethod
    def _test_path(value):
        """Test and convert input argument to pathlib.Path.

        If variable is not defined in environment variable, the default value
        is used.

        Args:
            value (`obj`): Argument to be tested

        """
        try:
            return Path(value)
        except TypeError:
            raise AttributeError('Not compatible with pathlib.Path')


path_var = GlobalPathVariablesManager()
