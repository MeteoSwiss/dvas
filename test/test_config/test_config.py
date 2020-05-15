"""
This file contains testing classes and function for
dvas.config.config module.

"""

# Import from python packages and modules
from pathlib import Path
from jsonschema import validate
import pytest

from dvas.config.config import instantiate_config_managers
from dvas.config.config import OrigData, Instrument, InstrType
from dvas.config.config import Parameter, Flag
from dvas.config.config import ConfigReadError
from dvas.dvas_environ import path_var as env_path_var

# Define
current_pkge_path = Path(__file__).parent
ok_fixture_dir = current_pkge_path / 'etc' / 'ok'
ko_fixture_dir = current_pkge_path / 'etc' / 'ko'


@pytest.mark.datafiles(ok_fixture_dir.as_posix())
def test_instantiate_config_managers(datafiles):
    """Test ConfigManager for OK config file"""

    cfg_mngrs_class = [OrigData, Instrument, InstrType, Parameter, Flag]

    # Instantiate all managers
    with env_path_var.set_many_attr(
        {'config_dir_path': Path(datafiles)}
    ):
        # Get managers
        cfg_mngrs = instantiate_config_managers(*cfg_mngrs_class, read=False)

        # Test type
        assert isinstance(cfg_mngrs, dict)

        # Loop for each manager
        for key, cfg_mngr in cfg_mngrs.items():
            # Test read
            assert cfg_mngr.read() is None

    # Test attribute exception
    with env_path_var.set_many_attr(
            {'config_dir_path': 'myfakepath'}
    ):
        # Get managers
        cfg_mngrs = instantiate_config_managers(*cfg_mngrs_class, read=False)

        with pytest.raises(AssertionError):
            # Loop for each manager
            for key, cfg_mngr in cfg_mngrs.items():
                cfg_mngr.read()


@pytest.mark.datafiles(Path.as_posix(ko_fixture_dir / 'a'))
def test_config_manager_ko_a(datafiles):
    """Test for bad config file"""

    with env_path_var.set_many_attr(
            {'config_dir_path': Path(datafiles)}
    ):
        # Instantiate all managers
        cfg_mngr = OrigData()

        with pytest.raises(ConfigReadError):
            cfg_mngr.read()


@pytest.mark.datafiles(Path.as_posix(ko_fixture_dir / 'b'))
def test_config_manager_ko_b(datafiles):

    with env_path_var.set_many_attr(
            {'config_dir_path': Path(datafiles)}
    ):
        # Instantiate all managers
        cfg_mngr = OrigData()

        with pytest.raises(ConfigReadError):
            cfg_mngr.read()
