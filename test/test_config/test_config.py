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

# Define
current_pkge_path = Path(__file__).parent
ok_fixture_dir = current_pkge_path / 'etc' / 'ok'
ko_fixture_dir = current_pkge_path / 'etc' / 'ko'


@pytest.mark.datafiles(ok_fixture_dir.as_posix())
def test_instantiate_config_managers(datafiles):
    """Test ConfigManager for OK config file"""

    cfg_mngrs = [OrigData, Instrument, InstrType, Parameter, Flag]

    # Instantiate all managers
    assert isinstance(instantiate_config_managers(cfg_mngrs, Path(datafiles).as_posix(), read=False), dict)
    assert isinstance(instantiate_config_managers(cfg_mngrs, Path(datafiles), read=False), dict)

    # Test attribute exception
    with pytest.raises(AssertionError):
        instantiate_config_managers(cfg_mngrs, 'myfakepath', read=False)
    with pytest.raises(AssertionError):
        instantiate_config_managers(cfg_mngrs, Path('myfakepath'), read=False)

    # Get managers
    cfg_mngrs = instantiate_config_managers(cfg_mngrs, Path(datafiles).as_posix(), read=False)

    # Loop for each manager
    for key, cfg_mngr in cfg_mngrs.items():

        # Test read
        assert cfg_mngr.read() is None


@pytest.mark.datafiles(Path.as_posix(ko_fixture_dir / 'a'))
def test_config_manager_ko_a(datafiles):
    """Test for bad config file"""

    # Instantiate all managers
    cfg_mngr = OrigData(Path(datafiles))

    with pytest.raises(ConfigReadError):
        cfg_mngr.read()


@pytest.mark.datafiles(Path.as_posix(ko_fixture_dir / 'b'))
def test_config_manager_ko_b(datafiles):

    # Instantiate all managers
    cfg_mngr = OrigData(Path(datafiles))

    with pytest.raises(ConfigReadError):
        cfg_mngr.read()
