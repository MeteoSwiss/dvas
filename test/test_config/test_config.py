"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

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
from dvas.config.config import OrigData, Instrument, InstrType
from dvas.config.config import Parameter, Flag, Tag
from dvas.config.config import ConfigReadError
from dvas.config.config import OneDimArrayConfigLinker
from dvas.config.config import ConfigGeneratorExpr
from dvas.config.config import ConfigGenMaxLenError
from dvas.environ import glob_var


def test_instantiate_config_managers():
    """Test ConfigManager for test_examples config file

    Tests:
        - Reading config files
        - Test reading different file name format and in sub dir

    """

    # Define
    cfg_mngrs_class = [OrigData, Instrument, InstrType, Parameter, Flag, Tag]

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

    # Test content for InstrType. Must contain exactly tst_list items (one time each one
    # This list will need to be udpated the if the test database changes.
    tst_list = ['AR-GDP_001', 'BR-GDP_001', 'RS41-GDP-BETA_001', 'YT', 'ZT', 'RS92', '']
    assert (sum(
        [(arg['type_name'] in tst_list) for arg in cfg_mngrs['InstrType']]
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

    cfg = InstrType()

    def test_read(self):
        """Test read method"""

        assert self.cfg.read("- type_name: 'TEST'\n  'desc': 'Test'") is None
        assert isinstance(self.cfg.document, list)

        # Test for bad field name (replace type_name by type)
        with pytest.raises(ConfigReadError):
            self.cfg.read("- type: 'TEST'\n  'desc': 'Test'")

        # Test for bad field type
        with pytest.raises(ConfigReadError):
            self.cfg.read("type_name: 'TEST'\ndesc': 'Test'")


class TestOneDimArrayConfigLinker:
    """Test OneDimArrayConfigLinker class

    Tests:
        - String generator
        - Catch generated string

    """

    # String generator patern
    prm_pat = re.compile(r'^dummytst_(param\d+)$')

    # Catched string patern
    desc_pat = re.compile(r'^param\d+$')

    # Config linker
    cfg_lkr = OneDimArrayConfigLinker([Parameter])

    def test_get_document(self):
        """Test get_document method

        Test:
            Returned type
            Item generator
            Raises ConfigGenMaxLenError

        """

        doc = self.cfg_lkr.get_document(Parameter.CLASS_KEY)

        assert isinstance(doc, list)
        assert sum(
            [self.prm_pat.match(arg['prm_abbr']) is not None for arg in doc]
        ) == 10
        assert sum(
            [self.desc_pat.match(arg['prm_desc']) is not None for arg in doc]
        ) == 10
        with glob_var.set_many_attr({'config_gen_max': 2}):
            with pytest.raises(ConfigGenMaxLenError):
                self.cfg_lkr.get_document(Parameter.CLASS_KEY)


class TestConfigGeneratorExpr:
    """Test ConfigGeneratorExpr class"""

    match_val = re.match(r'^a(\d)', 'a1')

    def test_eval(self):

        # Test str
        assert ConfigGeneratorExpr.eval("a", self.match_val) == 'a'

        # Test cat
        assert ConfigGeneratorExpr.eval("cat('a', 'b')", self.match_val) == 'ab'

        # Test replace
        assert ConfigGeneratorExpr.eval("rpl({'b': 'a'}, 'b')", self.match_val) == \
            ConfigGeneratorExpr.eval("repl({'b': 'a'}, 'b')", self.match_val) == \
            ConfigGeneratorExpr.eval("repl({'b': 'a'}, 'a')", self.match_val) == \
            'a'

        # Test replace strict
        assert ConfigGeneratorExpr.eval("rpls({'b': 'a'}, 'a')", self.match_val) == ''
        assert ConfigGeneratorExpr.eval("rpls({'b': 'a'}, 'b')", self.match_val) == \
            ConfigGeneratorExpr.eval("repl_strict({'b': 'a'}, 'b')", self.match_val) == \
            'a'

        # Test get
        assert ConfigGeneratorExpr.eval("get(1)", self.match_val) == '1'
        assert ConfigGeneratorExpr.eval("get(0)", self.match_val) == 'a1'

        # Test upper
        assert ConfigGeneratorExpr.eval("upper('a')", self.match_val) == 'A'

        # Test lower
        assert ConfigGeneratorExpr.eval("lower('A')", self.match_val) == 'a'

        # Test small upper
        assert ConfigGeneratorExpr.eval("supper('aa')", self.match_val) == \
            ConfigGeneratorExpr.eval("small_upper('aa')", self.match_val) == 'Aa'
