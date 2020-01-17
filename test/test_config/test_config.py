"""Testing module for .config.config

"""

# Import python packages and modules
import os
import re
from pathlib import Path
from jsonschema import validate
import pytest
from mdtpyhelper.check import CheckfuncAttributeError

from pampy import match, ANY

from uaii2021.config import CONST_KEY_NM, CONFIG_ITEM_PATTERN
from uaii2021.config.config import IdentifierManager
from uaii2021.config.config import ConfigManager
from uaii2021.config.config import ConfigReadError
from uaii2021.config.config import ConfigItemKeyError
from uaii2021.config.config import ConfigNodeError

# Define
ITEM_OK = {
    'raw_data': {
        'type_11.const.delimiter': ';',
        'type_11.instr_22.flight_33.batch_4.const.x_a': 3.0,
        'type_11.instr_22.flight_33.batch_4.const.T_a': 3.0,
    },
    'quality_check': {
        'flight_00.const.idx': [[1, 2], [1, 3]],
        'flight_00': {
            'const': {
                'idx': [[1, 2], [1, 3]], 'rep_param': 'T', 'rep_val': 88.1
            }
        }
    }
}
ITEM_KO = {
    'raw_data': [
        '99zz',
        'const.99zz',
        'type_11.instr_3',
    ],
    'quality_check': [
        '99zz',
        'const.99zz',
    ],
}

OK_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'etc/ok',
    )
KO_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'etc/ko',
    )


@pytest.mark.datafiles(OK_FIXTURE_DIR)
def test_config_manager_ok(datafiles):
    """Test ConfigManager for OK config file"""

    # Test attribute exception
    with pytest.raises(CheckfuncAttributeError):
        ConfigManager.instantiate_all_childs('mytutupath')

    # Instantiate all managers
    cfg_mngrs = ConfigManager.instantiate_all_childs(Path(datafiles))

    # Loop for each manager
    for key, cfg_mngr in cfg_mngrs.items():

        # Check constant attributes
        assert match(
            cfg_mngr.PARAMETER_SCHEMA, {"type": "object", "patternProperties": ANY, "additionalProperties": False},
            True, default=False) is True
        assert match(
            cfg_mngr.ROOT_PARAMS_DEF, {CONST_KEY_NM: ANY},
            True, default=False) is True

        # Test default root parameter
        assert validate(instance=cfg_mngr.ROOT_PARAMS_DEF, schema=cfg_mngr.json_schema) is None

        # Test read
        assert cfg_mngr.read() is None

        # Test item OK
        for itm_key, value in ITEM_OK[key].items():
            assert cfg_mngr[itm_key] == value
            assert cfg_mngr[itm_key.split('.')] == value

        # Test item KO
        for itm_key in ITEM_KO[key]:

            with pytest.raises(ConfigItemKeyError):
                cfg_mngr[itm_key]

            with pytest.raises(CheckfuncAttributeError):
                cfg_mngr[0]

            with pytest.raises(CheckfuncAttributeError):
                cfg_mngr['']


@pytest.mark.datafiles(os.path.join(KO_FIXTURE_DIR, 'a'))
def test_config_manager_ko_a(datafiles):
    """Test for bad config file"""

    # Instantiate all managers
    cfg_mngrs = ConfigManager.instantiate_all_childs(Path(datafiles))

    # Loop for each manager
    for key, cfg_mngr in cfg_mngrs.items():

        with pytest.raises(ConfigReadError):
            cfg_mngr.read()


@pytest.mark.datafiles(os.path.join(KO_FIXTURE_DIR, 'b'))
def test_config_manager_ko_b(datafiles):

    # Instantiate all managers
    cfg_mngrs = ConfigManager.instantiate_all_childs(Path(datafiles))

    # Loop for each manager
    for key, cfg_mngr in cfg_mngrs.items():

        with pytest.raises(ConfigReadError):
            cfg_mngr.read()


class TestIdentifierManager:

    # Instantiate test class
    inst = IdentifierManager(['flight', 'batch', 'ms'])

    def test_check_dict_nodes(self):

        end_node_pat = r'const'

        for arg in [re.compile(end_node_pat), end_node_pat]:
            assert self.inst.check_dict_node(
                {'const': 2,
                 'flight_0': {'const': 1},
                 'flight_1': {'batch_0': {'const': 1},
                              'batch_1': {'ms_0': {'const': 2}}}},
                arg
            ) is None

        with pytest.raises(Exception):
            self.inst.check_dict_node(
                {'flight_0': {'const': 1},
                 'flight_1': {'tutu': {'const': 1}}},
                end_node_pat
            )

        with pytest.raises(Exception):
            self.inst.check_dict_node(
                {'flight_0': {'const': 1},
                 'batch_1': {'ms_1': {'const': 1}}},
                end_node_pat
            )

        with pytest.raises(Exception):
            self.inst.check_dict_node(
                {'flight_0': {'const': 1},
                 'flight_0': {'batch_1': {'ms_1': 1}}},
                end_node_pat
            )
