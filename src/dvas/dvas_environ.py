"""
This module contains the package's environment variables.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import Python packages and module
import os

#: str: Environment variable used to define orig data directory path
ORIG_DATA_PATH_NM = 'DVAS_ORIG_DATA_PATH'
orig_data_path = os.getenv(ORIG_DATA_PATH_NM)

#: str: Environment variable used to define package's config directory path
CONFIG_PATH_NM = 'DVAS_CONFIG_PATH'
config_dir_path = os.getenv(CONFIG_PATH_NM)

#: str: Environment variable used to define package's local DB directory path
LOCAL_DB_PATH_NM = 'DVAS_LOCAL_DB_PATH'

#: str: Environment variable used to define output directory path
OUTPUT_PATH_NM = 'DVAS_OUTPUT_PATH'
