"""Testing module for .config.config

"""

# Import python packages and modules
import os
import re
from pathlib import Path
from jsonschema import validate
import pytest
from mdtpyhelper.check import CheckfuncAttributeError
import numpy as np

from pampy import match, ANY

from uaii2021.config import ITEM_SEPARATOR
from uaii2021.config import CONST_KEY_NM, CONFIG_ITEM_PATTERN
from uaii2021.config.config import IdentifierManager
from uaii2021.config.config import ConfigManager
from uaii2021.config.config import ConfigReadError
from uaii2021.config.config import ConfigItemKeyError
from uaii2021.config.config import IDNodeError


def test_init_const():
    """Test init.py"""

    test_str_OK = [
        'qc0',
        'qc1',
        '2010-01-20T001020Z',
        '2022-12-31T235959Z',
        'tre200s0',
        'PAY',
        'PAY_2',
        'PAY_RS',
        'PAY_L1',
        'vai-rs41',
        'metlab-c50',
        'b0',
        'f01',
        'i12']

    test_str_KO = [
        'qc-0',
        '2010-1-20T001020Z',
        '2022-1231T235959Z',
        '2022-12-31T2359Z',
        'tre200s',
        'T',
        'PAY_',
        'PAY2',
        'vairs41',
        'b_0',
        'f1',
        'i_12']

    for test_str in test_str_OK:
        assert len(
            [(key, test_str) for key, arg in CONFIG_ITEM_PATTERN.items()
             if re.match(arg, test_str) is None]) == (len(CONFIG_ITEM_PATTERN) - 1)

    for test_str in test_str_KO:
        assert len(
            [(key, test_str) for key, arg in CONFIG_ITEM_PATTERN.items()
             if re.match(arg, test_str) is None]) == len(CONFIG_ITEM_PATTERN)


# Define
ITEM_OK = {
    'raw_data': {
        f'vai-rs41{ITEM_SEPARATOR}const{ITEM_SEPARATOR}delimiter': ';',
        f'vai-rs41{ITEM_SEPARATOR}i22{ITEM_SEPARATOR}const{ITEM_SEPARATOR}delimiter': '.',
        f'vai-rs92{ITEM_SEPARATOR}i11{ITEM_SEPARATOR}const.x_a': 3.0,
        f'vai-rs92{ITEM_SEPARATOR}i11{ITEM_SEPARATOR}const{ITEM_SEPARATOR}trepros1_a': 3.0,
    },
    'quality_check': {
        f'f00{ITEM_SEPARATOR}const{ITEM_SEPARATOR}idx_val': [[1, 2], [1, 3]],
        f'f10{ITEM_SEPARATOR}b1{ITEM_SEPARATOR}const{ITEM_SEPARATOR}rep_val': np.nan,
        'f00': {
            'const': {
                'idx_val': [[1, 2], [1, 3]], 'rep_param': 'trepros1', 'rep_val': 88.1
            }
        }
    }
}
ITEM_KO = {
    'raw_data': [
        '99zz',
        f'const{ITEM_SEPARATOR}99zz',
        f'vai-rs92{ITEM_SEPARATOR}i3',
    ],
    'quality_check': [
        '99zz',
        f'const{ITEM_SEPARATOR}99zz',
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
    with pytest.raises(AssertionError):
        ConfigManager.instantiate_all_childs('mytutupath')
    with pytest.raises(AssertionError):
        ConfigManager.instantiate_all_childs(Path('mytutupath'))

    # Instantiate all managers
    cfg_mngrs = ConfigManager.instantiate_all_childs(Path(datafiles).as_posix())
    cfg_mngrs = ConfigManager.instantiate_all_childs(Path(datafiles))

    # Loop for each manager
    for key, cfg_mngr in cfg_mngrs.items():

        # Check constant attributes
        assert match(
            cfg_mngr.parameter_schema, {"type": "object", "patternProperties": ANY, "additionalProperties": False},
            True, default=False) is True
        assert match(
            cfg_mngr.root_params_def, {ANY: ANY},
            True, default=False) is True

        # Test default root parameter
        assert validate(instance=cfg_mngr.root_params_def, schema=cfg_mngr.parameter_schema) is None

        # Test read
        assert cfg_mngr.read() is None

        # Test item OK
        for itm_key, value in ITEM_OK[key].items():
            # Test get item
            assert cfg_mngr[itm_key] in [value]
            assert cfg_mngr[cfg_mngr._id_mngr.split(itm_key)] in [value]

            # Check all
            node = cfg_mngr._id_mngr.split(itm_key)
            try:
                i = node.index('const')
                node = node[:i]

                all_prm = cfg_mngr.get_all(node)
                assert isinstance(all_prm, dict)
                assert value in all_prm.values()

            except ValueError:
                pass

        # Test item KO
        for itm_key in ITEM_KO[key]:
            with pytest.raises(ConfigItemKeyError):
                cfg_mngr[itm_key]


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

    with pytest.raises(AssertionError):
        IdentifierManager(['tutu_test'])

    def test_get_item_id(self):

        assert self.inst.get_item_id(['f00', 'b1', 'PAY'], 'ms') == 'PAY'
        assert self.inst.get_item_id(['f00', 'b1', 'PAY'], 'flight') == 'f00'

        with pytest.raises(IDNodeError):
            self.inst.get_item_id(['i00', 'b1', 'PAY'], 'flight')

        with pytest.raises(IDNodeError):
            self.inst.get_item_id(['b1', 'PAY'], 'flight')

    def test_check_dict_nodes(self):

        end_node_pat = r'const'

        for arg in [re.compile(end_node_pat), end_node_pat]:
            assert self.inst.check_dict_node(
                {'const': 2,
                 'f00': {'const': 1},
                 'f11': {'b0': {'const': 1},
                         'b1': {'PAY': {'const': 2}}}},
                arg
            ) is None

        with pytest.raises(IDNodeError):
            self.inst.check_dict_node(
                {'f00': {'const': 1},
                 'f11': {'tutu': {'const': 1}}},
                end_node_pat
            )

        with pytest.raises(IDNodeError):
            self.inst.check_dict_node(
                {'f00': {'const': 1},
                 'b1': {'PAY': {'const': 1}}},
                end_node_pat
            )

        with pytest.raises(IDNodeError):
            self.inst.check_dict_node(
                {'f00': {'const': 1},
                 'f11': {'b1': {'PAY': 1}}},
                end_node_pat
            )

    def test_split(self):

        assert isinstance(self.inst.split('a.b'), list)
        assert isinstance(self.inst.split(['a', 'b']), list)

    def test_join(self):

        assert isinstance(self.inst.join('a.b'), str)
        assert isinstance(self.inst.join(['a', 'b']), str)
