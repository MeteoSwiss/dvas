"""
This module contains the package's environment variables.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import Python packages and module
from pathlib import Path

# Define package path
package_path = Path(__file__).parent

#: pathlib.Path: Original data directory path
orig_data_path = package_path / 'examples' / 'data'

#: pathlib.Path: Config directory path
config_dir_path = package_path / 'examples' / 'config'

#: pathlib.Path: Local DB directory path
local_db_path = Path('.') / 'dvas_db'

#: pathlib.Path: Output directory path
output_path = Path('.') / 'output'
