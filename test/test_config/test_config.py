"""Testing module for .config.config

"""

# Import python packages and modules
import os
import re
from pathlib import Path
from jsonschema import validate
import pytest
import numpy as np

from pampy import match, ANY

from uaii2021.config import ITEM_SEPARATOR
from uaii2021.config import CONST_KEY_NM, CONFIG_ITEM_PATTERN
from uaii2021.config.config import ConfigManager, RawData
from uaii2021.config.config import ConfigReadError
from uaii2021.config.config import ConfigItemKeyError
from uaii2021.config.config import IDNodeError


def test_init_const():
    """Test init.py"""

    test_str_OK = [
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
    'raw_data': [
        {'key': ['vai-rs41', 'delimiter'], 'value': ';'},
        {'key': ['vai-rs41', 'i22', 'delimiter'], 'value': '.'},
        {'key': ['vai-rs92', 'i11', 'x_func'], 'value': 'lambda x: 3 * x'},
        {'key': ['vai-rs92', 'i11', 'trepros1_func'], 'value': 'lambda x: 3 * x'},
    ],
    'instr_type': [
        {'key': ['instr_type_0', 'name'], 'value': 'vai-rs41'}
    ],
    'instrument': [
        {'key': ['instrument_0', 'id'], 'value': 'i0'}
    ],
    'flight': [
        {'key': ['flight_0', 'batch_amount'], 'value': 2}
    ]
}
ITEM_KO = [
    ['myfakeitem'],
    ['flight_0'],
    ['instrument_0', 'myfakeitem']
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
def test_instantiate_all_child(datafiles):
    """Test ConfigManager for OK config file"""

    # Instantiate all managers
    assert isinstance(ConfigManager.instantiate_all_childs(Path(datafiles).as_posix(), read=False), dict)
    assert isinstance(ConfigManager.instantiate_all_childs(Path(datafiles), read=False), dict)

    # Test attribute exception
    with pytest.raises(AssertionError):
        ConfigManager.instantiate_all_childs('myfakepath', read=False)
    with pytest.raises(AssertionError):
        ConfigManager.instantiate_all_childs(Path('myfakepath'), read=False)

    # Get managers
    cfg_mngrs = ConfigManager.instantiate_all_childs(Path(datafiles).as_posix(), read=False)

    # Loop for each manager
    for key, cfg_mngr in cfg_mngrs.items():

        # Test root parameter
        assert type(cfg_mngr.root_params_def) is dict
        assert type(cfg_mngr.parameter_pattern_prop) is dict

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",

            "type": 'object',
            "patternProperties": cfg_mngr.parameter_pattern_prop,
            "additionalProperties": False
        }

        assert validate(instance=cfg_mngr.root_params_def, schema=schema) is None

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
    cfg_mngr = RawData(Path(datafiles))

    with pytest.raises(ConfigReadError):
        cfg_mngr.read()


@pytest.mark.datafiles(os.path.join(KO_FIXTURE_DIR, 'b'))
def test_config_manager_ko_b(datafiles):

    # Instantiate all managers
    cfg_mngr = RawData(Path(datafiles))

    with pytest.raises(ConfigReadError):
        cfg_mngr.read()


#TODO
# Delete this test

# class TestIdentifierManager:
#
#     # Instantiate test class
#     inst = IdentifierManager(['flight', 'batch', 'ms'])
#
#     with pytest.raises(AssertionError):
#         IdentifierManager(['tutu_test'])
#
#     def test_get_item_id(self):
#
#         assert self.inst.get_item_id(['f00', 'b1', 'PAY'], 'ms') == 'PAY'
#         assert self.inst.get_item_id(['f00', 'b1', 'PAY'], 'flight') == 'f00'
#
#         with pytest.raises(IDNodeError):
#             self.inst.get_item_id(['i00', 'b1', 'PAY'], 'flight')
#
#         with pytest.raises(IDNodeError):
#             self.inst.get_item_id(['b1', 'PAY'], 'flight')
#
#     def test_check_dict_nodes(self):
#
#         end_node_pat = r'const'
#
#         for arg in [re.compile(end_node_pat), end_node_pat]:
#             assert self.inst.check_dict_node(
#                 {'const': 2,
#                  'f00': {'const': 1},
#                  'f11': {'b0': {'const': 1},
#                          'b1': {'PAY': {'const': 2}}}},
#                 arg
#             ) is None
#
#         with pytest.raises(IDNodeError):
#             self.inst.check_dict_node(
#                 {'f00': {'const': 1},
#                  'f11': {'tutu': {'const': 1}}},
#                 end_node_pat
#             )
#
#         with pytest.raises(IDNodeError):
#             self.inst.check_dict_node(
#                 {'f00': {'const': 1},
#                  'b1': {'PAY': {'const': 1}}},
#                 end_node_pat
#             )
#
#         with pytest.raises(IDNodeError):
#             self.inst.check_dict_node(
#                 {'f00': {'const': 1},
#                  'f11': {'b1': {'PAY': 1}}},
#                 end_node_pat
#             )
#
#     def test_split(self):
#
#         assert isinstance(self.inst.split('a.b'), list)
#         assert isinstance(self.inst.split(['a', 'b']), list)
#
#     def test_join(self):
#
#         assert isinstance(self.inst.join('a.b'), str)
#         assert isinstance(self.inst.join(['a', 'b']), str)
