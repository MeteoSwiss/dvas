"""
This file contains testing classes and function for dvas.config.config modules

"""

# Import from python packages and modules
import os
from pathlib import Path
from jsonschema import validate
import pytest

from dvas.config.config import instantiate_config_managers
from dvas.config.config import OrigData, Instrument, InstrType
from dvas.config.config import Parameter, Flag
from dvas.config.config import ConfigReadError
from dvas.config.config import ConfigItemKeyError
from dvas.config.pattern import INSTR_TYPE_KEY, INSTR_KEY
from dvas.config.pattern import EVENT_KEY, PARAM_KEY
from dvas.config.pattern import FLAG_KEY
from dvas.config.pattern import ORIGDATA_KEY

# Define
ITEM_OK = {
    ORIGDATA_KEY: [
        {'key': ['vai-rs41', 'delimiter'], 'value': ';'},
        {'key': ['vai-rs41', 'trepros1', 'delimiter'], 'value': '.'},
        {'key': ['vai-rs92', 'trepros1', 'lambda'], 'value': 'lambda x: x'},
        {'key': ['vai-rs92', 'trepros1', 'i11', 'lambda'], 'value': 'lambda x: 3 * x'},
    ],
    INSTR_TYPE_KEY: [
        {'key': ['instr_type_1', 'name'], 'value': 'vai-rs41'}
    ],
    INSTR_KEY: [
        {'key': ['instrument_1', 'instr_id'], 'value': 'i1'},
        {'key': ['instrument_2', 'instr_id'], 'value': 'i2'}
    ],
    PARAM_KEY: [
        {'key': ['parameter_1', 'prm_abbr'], 'value': 'treprot1'}
    ],
    FLAG_KEY: [
        {'key': ['flag_0', 'bit_number'], 'value': 0}
    ]
}
ITEM_KO = [
    ['myfakeitem'],
    ['flight_1'],
    ['instrument_1', 'myfakeitem']
]

OK_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'etc/ok',
    )
KO_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'etc/ko',
    )


@pytest.mark.datafiles(OK_FIXTURE_DIR)
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

        # # Test root parameter
        # assert type(cfg_mngr.root_params_def) is dict
        # assert type(cfg_mngr.parameter_pattern_prop) is dict
        #
        # schema = {
        #     "$schema": "http://json-schema.org/draft-07/schema#",
        #
        #     "type": 'object',
        #     "patternProperties": cfg_mngr.root_params_pattern_prop,
        #     "additionalProperties": False
        # }
        #
        # assert validate(instance=cfg_mngr.root_params_def, schema=schema) is None
        #
        # # Test const node field
        # schema = cfg_mngr.json_schema.copy()
        # schema["nodeItem"]["patternProperties"].update(cfg_mngr.root_params_pattern_prop)
        # schema["nodeItem"]["patternProperties"].update(
        #     {rf"^(({')('.join(cfg_mngr.const_nodes.keys())}))$": {"$ref": "#/nodeItem"}}
        # )
        #
        # assert validate(instance=cfg_mngr.const_nodes, schema=schema) is None

        # Test read
        assert cfg_mngr.read() is None

        # Test item OK
        for arg in ITEM_OK[key]:
            # Test get item
            assert cfg_mngr[arg['key']] in [arg['value']]

            # Check all
            try:
                i = arg['key'].index('const')
                node = arg['key'][:i]

                all_prm = cfg_mngr.get_all(node)
                assert isinstance(all_prm, dict)
                assert arg['value'] in all_prm.values()

            except ValueError:
                pass

        # Test item KO
        for itm_key in ITEM_KO:
            with pytest.raises(ConfigItemKeyError):
                cfg_mngr[itm_key]


@pytest.mark.datafiles(os.path.join(KO_FIXTURE_DIR, 'a'))
def test_config_manager_ko_a(datafiles):
    """Test for bad config file"""

    # Instantiate all managers
    cfg_mngr = OrigData(Path(datafiles))

    with pytest.raises(ConfigReadError):
        cfg_mngr.read()


@pytest.mark.datafiles(os.path.join(KO_FIXTURE_DIR, 'b'))
def test_config_manager_ko_b(datafiles):

    # Instantiate all managers
    cfg_mngr = OrigData(Path(datafiles))

    with pytest.raises(ConfigReadError):
        cfg_mngr.read()
