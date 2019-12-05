"""Testing module for .config.config

"""

# Import current package module
from uaii2021.config.rawdata import ROOT_PARAMS_DEF
from uaii2021.config.rawdata import PARAMETER_SCHEMA


def test_define():

    assert type(ROOT_PARAMS_DEF) is dict
    assert type(PARAMETER_SCHEMA) is dict
