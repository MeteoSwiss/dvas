"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing classes and function for dvas.config.config module.

"""

# Import from python packages and modules
import re
import pytest

# Import from current python packages and modules
from dvas.config.config import instantiate_config_managers
from dvas.config.config import CSVOrigMeta
from dvas.config.config import OrigData, Model
from dvas.config.config import Parameter, Flag, Tag
from dvas.config.config import ConfigReadError, ConfigGetError
from dvas.config.config import ConfigExprInterpreter
from dvas.config.config import ExprInterpreterError
from dvas.environ import glob_var
from dvas.hardcoded import FLAG_PRM_NAME_SUFFIX


def test_instantiate_config_managers():
    """Test ConfigManager for test_examples config file

    Tests:
        - Reading config files
        - Test reading different file name format and in sub dir

    """

    # Define
    cfg_mngrs_class = [OrigData, Model, Parameter, Flag, Tag]

    # Instantiate all managers
    cfg_mngrs = instantiate_config_managers(*cfg_mngrs_class, read=False)

    # Test type
    assert isinstance(cfg_mngrs, dict)
    assert all([arg.CLASS_KEY in cfg_mngrs.keys() for arg in cfg_mngrs_class])
    assert all([isinstance(arg, tuple(cfg_mngrs_class)) for arg in cfg_mngrs.values()])

    # Read each manager
    cfg_mngrs = instantiate_config_managers(*cfg_mngrs_class, read=True)
    for cfg_mngr in cfg_mngrs.values():

        # Test read
        assert cfg_mngr.document is not None

    # Test content for Model. Must contain exactly tst_list items (one time each one
    # This list will need to be udpated the if the test database changes.
    tst_list = ['AR-GDP_001', 'BR-GDP_001', 'RS41-GDP-BETA_001', 'YT', 'ZT', 'RS92']
    assert (sum(
        [(arg['mdl_name'] in tst_list) for arg in cfg_mngrs['Model']]
    ) == len(tst_list))


class TestOneLayerConfigManager():
    """Test class for OneLayerConfigManager

    Test:
        - Read a right YAML formatted str
        - Raise for bad field type

    """

    # Define
    cfg = CSVOrigMeta()

    def test_read(self):
        """Test read method"""

        assert self.cfg.read("field1: 'a'\nfield2: 'b'") is None
        assert isinstance(self.cfg.document, dict)

        # Test for bad field type
        with pytest.raises(ConfigReadError):
            self.cfg.read("field1:\n- 'a'\n- 'b'")


class TestOneDimArrayConfigManager():
    """Test class for OneDimArrayConfigManager

    Test:
    - Read a right YAML formatted str
    - Raise for bad field name
    - Raise for bad field type

    """

    cfg = Model()

    def test_read(self):
        """Test read method"""

        assert self.cfg.read("- mdl_name: 'TEST'\n  'mdl_desc': 'Test'") is None
        assert isinstance(self.cfg.document, list)

        # Test for bad field name (replace mdl_name by type)
        with pytest.raises(ConfigReadError):
            self.cfg.read("- type: 'TEST'\n  'mdl_desc': 'Test'")

        # Test for bad field type
        with pytest.raises(ConfigReadError):
            self.cfg.read("mdl_name: 'TEST'\nmdl_desc': 'Test'")


    def test_get(self):
        """Test get methods (__getitem__)"""

        self.cfg.read("- mdl_name: 'TEST'\n  'mdl_desc': 'Test'")

        assert self.cfg[0]['mdl_name'] == 'TEST'

        # Test bad key
        with pytest.raises(ConfigGetError):
            self.cfg['mdl_name']

        with pytest.raises(IndexError):
            self.cfg[2]


class TestParameter:
    """Test class for Parameter"""

    # Dummy param number
    n_dummy = 10

    # String generator patern
    prm_pat = re.compile(r'dummytst_(param\d)')
    flg_pat = re.compile(rf'{prm_pat.pattern}{FLAG_PRM_NAME_SUFFIX}')

    # Catched string patern
    desc_pat = re.compile(r'param\d+')

    def test_read(self):
        """Test read method"""

        # Init
        cfg = Parameter()
        cfg.read(f"- prm_name: {self.prm_pat.pattern}\n  prm_desc: get(1)")

        assert sum(
            [self.prm_pat.fullmatch(arg['prm_name']) is not None for arg in cfg.document]
        ) == self.n_dummy
        assert sum(
            [self.flg_pat.fullmatch(arg['prm_name']) is not None for arg in cfg.document]
        ) == self.n_dummy
        assert sum(
            [self.desc_pat.fullmatch(arg['prm_desc']) is not None for arg in cfg.document]
        ) == self.n_dummy

        # Test gnerator limit overflow
        with glob_var.protect():
            glob_var.config_gen_max = 2
            with pytest.raises(ConfigReadError):
                cfg.read(f"- prm_name: {self.prm_pat.pattern}\n  prm_desc: get(1)")


class TestConfigExprInterpreter:
    """Test ConfigExprInterpreter class"""

    match_val = re.match(r'^a(\d)', 'a1')

    def test_set_callable(self):
        """Test set_method method"""

        assert ConfigExprInterpreter.set_callable(self.match_val.group) is None

        # Test raise
        with pytest.raises(AssertionError):
            ConfigExprInterpreter.set_callable('xxx')

    def test_eval(self):
        """Test eval method"""

        # Test str
        assert ConfigExprInterpreter.eval("a", self.match_val.group) == 'a'
        assert ConfigExprInterpreter.eval("get(1", self.match_val.group) == 'get(1'

        # Test cat
        assert ConfigExprInterpreter.eval("cat('a', 'b')", self.match_val.group) == 'ab'

        # Test replace
        assert ConfigExprInterpreter.eval("rpl({'b': 'a'}, 'b')", self.match_val.group) == \
            ConfigExprInterpreter.eval("repl({'b': 'a'}, 'b')", self.match_val.group) == \
            ConfigExprInterpreter.eval("repl({'b': 'a'}, 'a')", self.match_val.group) == \
            'a'

        # Test replace strict
        assert ConfigExprInterpreter.eval("rpls({'b': 'a'}, 'a')", self.match_val.group) == ''
        assert ConfigExprInterpreter.eval("rpls({'b': 'a'}, 'b')", self.match_val.group) == \
            ConfigExprInterpreter.eval("repl_strict({'b': 'a'}, 'b')", self.match_val.group) == \
            'a'

        # Test get
        assert ConfigExprInterpreter.eval("get(1)", self.match_val.group) == '1'
        assert ConfigExprInterpreter.eval("get(0)", self.match_val.group) == 'a1'

        # Test upper
        assert ConfigExprInterpreter.eval("upper('a')", self.match_val.group) == 'A'

        # Test lower
        assert ConfigExprInterpreter.eval("lower('A')", self.match_val.group) == 'a'

        # Test small upper
        assert ConfigExprInterpreter.eval("supper('aa')", self.match_val.group) == \
            ConfigExprInterpreter.eval("small_upper('aa')", self.match_val.group) == 'Aa'

        # Test error
        with pytest.raises(ExprInterpreterError):
            ConfigExprInterpreter.eval("supper(1)", self.match_val.group)

        with pytest.raises(ExprInterpreterError):
            ConfigExprInterpreter.eval("get(10)", self.match_val.group)
