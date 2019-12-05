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

from uaii2021.config import PARAM_KEY_NM, NODE_PAT_DICT
from uaii2021.config.config import ConfigManager
from uaii2021.config.config import ConfigReadError
from uaii2021.config.config import ConfigItemKeyError
from uaii2021.config.config import ConfigNodeError

# Define
ITEM_OK = {
    'raw_data': {
        'instr_type_11.param.delimiter': ';',
        'instr_type_11.instr_22.flight_33.batch_4.param.x_a': 3.0,
        'instr_type_11.instr_22.flight_33.batch_4.param.T_a': 3.0,
    },
    'quality_check': {
        'flight_00.param.idx': [[1, 2], [1, 3]],
        'flight_00': {
            'param': {
                'idx': [[1, 2], [1, 3]], 'rep_param': 'T', 'rep_val': 88.1
            }
        }
    }
}
ITEM_KO = {
    'raw_data': [
        '99zz',
        'param.99zz',
        'instr_type_11.instr_3',
    ],
    'quality_check': [
        '99zz',
        'param.99zz',
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
        ConfigManager.instantiate_all_childs('.')

    # Instantiate all managers
    cfg_mngrs = ConfigManager.instantiate_all_childs(Path(datafiles))

    # Loop for each manager
    for key, cfg_mngr in cfg_mngrs.items():

        # Check constant attributes
        assert match(
            cfg_mngr.PARAMETER_SCHEMA, {"type": "object", "patternProperties": ANY, "additionalProperties": False},
            True, default=False) is True
        assert match(
            cfg_mngr.ROOT_PARAMS_DEF, {PARAM_KEY_NM: ANY},
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


def test_check_dict_nodes():

    assert ConfigManager._check_dict_nodes(
        {'flight0': {'param': 1}, 'flight1': {'batch': {'param': 1}}},
        [re.compile(arg) for arg in [r'^flight\w$', '^batch$']],
        re.compile('^param$')
    ) is None

    assert ConfigManager._check_dict_nodes(
        {'flight0': {'param': 1}, 'flight1': {'batch': {'param': 1}}},
        [r'^flight\w$', '^batch$'],
        '^param$'
    ) is None

    with pytest.raises(ConfigNodeError):
        ConfigManager._check_dict_nodes(
            {'flight0': {'param': 1}, 'flight1': {'tutu': {'param': 1}}},
            [re.compile(arg) for arg in [r'^flight\w$', '^batch$']],
            re.compile('^param$')
        )

    with pytest.raises(ConfigNodeError):
        ConfigManager._check_dict_nodes(
            {'flight0': {'param': 1}, 'flight1': {'batch': {'tutu': {'param': 1}}}},
            [re.compile(arg) for arg in [r'^flight\w$', '^batch$']],
            re.compile('^param$')
        )

    with pytest.raises(ConfigNodeError):
        ConfigManager._check_dict_nodes(
            {'flight0': {'params': 1}, 'flight1': {'batch': {'param': 1}}},
            [re.compile(arg) for arg in [r'^flight\w$', '^batch$']],
            re.compile('^param$')
        )

def test_check_list_item():

    assert ConfigManager._check_list_item(
        ['flight0', 'batch'],
        [re.compile(arg) for arg in [r'^flight\w$', '^batch$']]
    ) is None

    assert ConfigManager._check_list_item(
        ['flight0'],
        [r'^flight\w$', '^batch$']
    ) is None

    with pytest.raises(ConfigItemKeyError):
        ConfigManager._check_list_item(
            ['flight'],
            [r'^flight\w$', '^batch$']
        )

    with pytest.raises(ConfigItemKeyError):
        ConfigManager._check_list_item(
            ['flight0', 'batch', 'tutu'],
            [r'^flight\w$', '^batch$']
        )